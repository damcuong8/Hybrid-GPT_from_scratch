
import unicodedata
import multiprocessing as mp
from cs336_basics.pretokenization_example import find_chunk_boundaries, process_chunk


def get_stats(ids, counts=None):
    """
    Given a list of intergers, return a dicttionary of counts of consecutive pairs
    Example = [1, 2, 3, 2, 3] -> {(1,2): 1, (2, 3): 2, (3,2):1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1

    return counts   

def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def merge_and_update(ids, pair, idx, stats):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    p0, p1 = pair
    is_pair = False
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if i < len(ids) - 1 and ids[i] == p0 and ids[i+1] == p1:
            # update stats relevent to previous token
            if len(newids) > 0:
                # old pair is impacted
                stats[(newids[-1], ids[i])] = stats.get((newids[-1], ids[i]), 0) - 1
                # new pair
                stats[(newids[-1], idx)] = stats.get((newids[-1], idx), 0) + 1
            # update stats relevent to next token
            if i < len(ids) - 2:
                stats[(p1, ids[i+2])] = stats.get((p1, ids[i+2]), 0) - 1
            newids.append(idx)
            i += 2
            is_pair = True
        else:
            if is_pair:
                stats[(idx, ids[i])] = stats.get((idx, ids[i]), 0) + 1
                is_pair = False
            newids.append(ids[i])
            i += 1
    return newids
    
class Tokenizer:
    " Base class "
    def __init__(self, vocab=None, merges=None, special_tokens=None):
        self.merges = merges if merges else {} # (int, int) -> int
        self.special_tokens = special_tokens if special_tokens else {} # str -> int
        self.vocab = vocab if vocab else self._build_vocab() # int -> bytes
        

    def train(self, input_path, vocab_size, verbose=False):

        raise NotImplementedError

    def _build_vocab(self):
        # raw bytes
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for idx, pair in self.merges.items():
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode('utf-8')

        return vocab      
    
    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()


class GPT2Tokenizer(Tokenizer):
    def __init__(self, special_tokens=None):
        super().__init__(special_tokens=special_tokens)
        self.inverse_special_tokens = {}

    def train(self, input_path, vocab_size, verbose=False):

        assert vocab_size >= 256
        num_merges = vocab_size - 256

        num_process = 4
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_process, "<|endoftext|>".encode("utf-8"))
            chunks = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode('utf-8', errors='ignore')
                chunks.append(chunk)

        with mp.Pool(num_process) as pool:
            results = pool.map(process_chunk, chunks)
        
        ids = [list(tok.encode('utf-8')) for sublist in results for tok in sublist]
        
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        stats = {}
        for chunk_ids in ids:
            stats = get_stats(chunk_ids, stats)
        for i in range(num_merges): 
            pair = max(stats, key=stats.get)

            idx = i + 256

            ids = [merge_and_update(chunk_ids, pair, idx, stats) for chunk_ids in ids]
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            print(pair)
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
            # remove stats pair
            del stats[pair]

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def _encode_chunk(chunk_bytes):
        ids = list(chunk_bytes)
        
        while len(ids) >= 2:
            stats = get_stats(ids)

            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break

            idx = self.merges[pair]

            ids = merge(ids, pair, idx)
        return ids


        

    def encode(self, text):
        text_chunks = process_chunk(text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode('utf-8')
            ids.extend(self._encode_chunk(chunk_bytes))
        
        return ids

    def decode(self, ids):
        # Given the list ids token return the string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx])
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")

        return text


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer()
    tokenizer.train("cs336_basics/TinyStoriesV2-GPT4-valid.txt", vocab_size=1000, verbose=True)


