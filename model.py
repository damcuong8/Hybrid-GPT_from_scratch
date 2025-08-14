from torch.utils.data import Dataset
from torch.amp import autocast
import torch
import torch.nn as nn
from torch import Tensor
import math
import logging
from collections.abc import Callable, Iterable
import typing
from typing import Optional, Union, Dict, Any, Tuple, List
import numpy as np
from tqdm import tqdm



logger = logging.getLogger(__name__)

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        std = (2 / (out_features + in_features))**0.5

        self.weight = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(out_features, in_features, **factory_kwargs),
            mean=0.0, std=std, a=-3*std, b=3*std)
            )
        
    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dims: int, device=None, dtype=None):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.embedding_table = nn.Parameter(torch.empty(num_embeddings, embedding_dims, **factory_kwargs))

        nn.init.trunc_normal_(self.embedding_table, mean=0.0, std=1, a=-3, b=3)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.embedding_table[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.eps = eps
        self.d_model = d_model

        factory_kwargs = {"device": device, "dtype": dtype}

        self.gi = nn.Parameter(torch.ones(self.d_model, **factory_kwargs))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

        x_norm = x * rms * self.gi

        return x_norm.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int=None, device=None, dtype=None):
        super().__init__()

        d_ff = d_ff if d_ff else int(math.ceil((8/3) * d_model / 64 ) * 64)

        factory_kwargs = {"device": device, "dtype": dtype}

        self.W_1 = Linear(d_model, d_ff, **factory_kwargs)
        self.W_3 = Linear(d_model, d_ff, **factory_kwargs)

        self.W_2 = Linear(d_ff, d_model, **factory_kwargs)

    def forward(self, x: Tensor) -> Tensor:

        x1 = self.W_1(x)

        swiglu = self.W_2(x1 * torch.sigmoid(x1) * self.W_3(x))

        return swiglu

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        inv_freq = torch.exp(-math.log(theta) * torch.arange(0, d_k, 2).float() / d_k)
        positions = torch.arange(max_seq_len).float()

        freqs = torch.outer(positions, inv_freq) # (max_seq_len, d_k/2)

        cos = freqs.cos()

        sin = freqs.sin()

        # Register as buffers (not learnable)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        # x: (..., seq_len, d_k)
        # token_positions: (..., seq_len)
        # Expand cos/sin using token positions
        cos_pos = self.cos[token_positions] # (..., seq_len, d_k/2)
        sin_pos = self.sin[token_positions] # (..., seq_len, d_k/2)

        x1, x2 = x[..., ::2], x[..., 1::2] # (..., seq_len, d_k/2) each

        x_rotated = torch.stack([
            x1 * cos_pos - x2 * sin_pos,
            x1 * sin_pos + x2 * cos_pos
        ], dim=-1) # (..., seq_len, d_k/2, 2)

        return x_rotated.flatten(-2)
    

def softmax(x: Tensor, dim: int):
    max_value = torch.amax(x, dim=dim, keepdim=True)
    x_stable = x - max_value

    exp_x = torch.exp(x_stable)

    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)

    return exp_x / sum_exp

def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor=None):

    d_k = q.size(-1)
    
    attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:

        attention_scores = attention_scores.masked_fill(mask == False, float("-inf"))

    attention = softmax(attention_scores, -1) # (batch_size, ..., seq_len, seq_len)

    context = attention @ v # (batch_size, ..., seq_len, d_v)

    return context

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int,
                num_heads: int,
                positional_encoder: RoPE):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.W_q = Linear(self.d_model, self.d_model)
        self.W_k = Linear(self.d_model, self.d_model)
        self.W_v = Linear(self.d_model, self.d_model)

        self.W_o = Linear(d_model, d_model)

        self.positional_encoder=positional_encoder

    def forward(self, 
                x: Tensor,
                causal_mask: Tensor=None,
                token_positions: Tensor=None) -> Tensor:

        *batch_dims, seq_len, d_model = x.size()
        num_heads = self.num_heads
        assert d_model == self.d_model
        
        # (batch_size, seq_len, d_model)
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
        q = q.view(*batch_dims, -1, num_heads, self.d_k).transpose(-3, -2)
        k = k.view(*batch_dims, -1, num_heads, self.d_k).transpose(-3, -2)
        v = v.view(*batch_dims, -1, num_heads, self.d_k).transpose(-3, -2)


        if token_positions is None:
            token_positions = torch.arange(seq_len).view(*(1,) * len(batch_dims), seq_len)

        token_positions = token_positions.unsqueeze(-2)

        q = self.positional_encoder(q, token_positions)
        k = self.positional_encoder(k, token_positions)

        if causal_mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
            causal_mask = mask.view(*(1,) * len(batch_dims), 1, seq_len, seq_len)

        context = scaled_dot_product_attention(q, k, v, mask=causal_mask)

        # (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, d_model)
        context = context.transpose(-3, -2).contiguous().view(*batch_dims, -1, num_heads*self.d_k) 

        output = self.W_o(context)

        return output

class TransformerBlock(nn.Module):
    def __init__(self,
                d_model: int,
                num_heads: int,
                d_ff: int,
                positional_encoder: RoPE):
        super().__init__()

        self.rms_norm_1 = RMSNorm(d_model)

        self.rms_norm_2 = RMSNorm(d_model)

        self.mha = MultiheadSelfAttention(d_model=d_model, num_heads=num_heads, positional_encoder=positional_encoder)

        self.feed_forward = SwiGLU(d_model, d_ff)

    def forward(self, x: Tensor, mask: Tensor=None):

        head = self.mha(self.rms_norm_1(x)) + x

        output = self.feed_forward(self.rms_norm_2(head)) + head

        return output


class TransformerLM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 num_layers: int,
                 num_heads:int,
                 d_model: int,
                 d_ff: int,
                 theta: int):
        super().__init__()

        self.context_length = context_length
        self.num_layers = num_layers
        self.final_layer_norm = RMSNorm(d_model=d_model)
        self.output_proj = Linear(in_features=d_model, out_features=vocab_size)

        self.embedding_table = Embedding(num_embeddings=vocab_size, embedding_dims=d_model)
        d_head = d_model // num_heads
        self.positional_encoder = RoPE(theta=theta, d_k=d_head, max_seq_len=context_length)

        self.transformer_layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    positional_encoder=self.positional_encoder
                ) 
                for _ in range(num_layers)
            ]
        )

        logger.info(f"number of non-embedding parameters: {self.get_num_params() / 1e6:.2f}M")
        
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the output_proj parameters get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.output_proj.weight.numel()

        return n_params
    

    def forward(self, token_ids: Tensor):
        x = self.embedding_table(token_ids)

        for i in range(self.num_layers):
            x = self.transformer_layers[i](x)

        output = self.output_proj(self.final_layer_norm(x))

        return output

    @torch.no_grad()
    def generate(
        self,
        x: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float | None = None,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ):
        """
        Args:
            x: LongTensor of shape `(1, sequence_length,)` or `(sequence_length, )`.
                Input IDs to condition on when generating.
            max_new_tokens: int
                Maximum number of tokens to generate.
            temperature: float
                Temperature to use during generation.
            top_k: int
                If provided, only sample from the `top_k` vocab items (by probability).
            eos_token_id: int
                If provided, stop generation when we generate this ID.

        Returns: A LongTensor of shape (max_new_tokens,) with the generated model output.
        """

        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        original_context_length = x.size(-1)

        for _ in range(max_new_tokens):
            # take the last context if the input is
            # beyond the model context length
            x = x[:,-self.context_length:] if original_context_length > self.context_length else x
            # get the logits
            logits = self.forward(x)
            
            next_logit_token = logits[:, -1]

            temperature_scaled_next_token_logits = next_logit_token / temperature

            if top_k is not None:
                topk_values = torch.topk(
                    temperature_scaled_next_token_logits,
                    min(top_k, temperature_scaled_next_token_logits.size(-1))
                )
                # Get the score of k elements and mask the other
                threshold = topk_values[:, -1]
                topk_mask = temperature_scaled_next_token_logits < threshold
                temperature_scaled_next_token_logits.masked_fill(topk_mask, float("-inf"))

            # If top-p is provided, apply top-p filtering
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(temperature_scaled_next_token_logits, descending=True, dim=-1)

                sorted_probs = softmax(sorted_logits, dim=-1)

                cum_probs = torch.cumsum(sorted_probs, dim=-1)

                top_p_mask = cum_probs > top_p
                # keep the first token surpass p threshold
                top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()
                
                top_p_mask[..., 0] = False

                sorted_logits.masked_fill(top_p_mask, float("-inf"))

                # move the sorted logits to the original position
                temperature_scaled_next_token_logits = sorted_logits.scatter(dim=-1, index=sorted_idx, src=sorted_logits)

            next_token_probabilities = softmax(temperature_scaled_next_token_logits, dim=-1)

            next_token_id = torch.multinomial(next_token_probabilities, 1)

            if eos_token_id is not None and eos_token_id == next_token_id.item():
                break
            x = torch.cat((x, next_token_id), dim=-1)
        
        new_token_ids = x[:, original_context_length:]
        return new_token_ids


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.detach()
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")
                
                state = self.state[p]

                # initial state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['step'] += 1
                t = state['step']

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                if group['weight_decay'] != 0.0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])

        return loss
        

def lr_cosine_schedule(t: int, a_max: float, a_min: float, T_w: int, T_c: int):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        t (int): Iteration number to get learning rate for.
        a_max (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        a_min (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        T_w (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        T_c (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    # warm up
    if t < T_w:
        a_t = (t / T_w) * a_max
    # Cosine annealing
    elif T_w <= t <= T_c:
        a_t = a_min + 1/2 * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (a_max - a_min)
    # Post-annealing
    else:
        a_t = a_min

    return a_t

def norm_gradient_clipping_(params: Iterable, max_l2_norm, eps=1e-6):
    
    total_norm = 0
    for p in params:
        if p.grad is not None:
            total_norm += p.grad.detach().pow(2).sum()
            
    total_norm = math.sqrt(total_norm + eps)

    if total_norm > max_l2_norm:
        coeffi_norm = max_l2_norm / (total_norm + eps)
        for p in params:
            if p.grad is not None:
                p.grad.detach().mul_(coeffi_norm)


def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str):

    max_start = len(x) - context_length

    starts = np.random.randint(0, max_start, size=batch_size)

    input_batch = np.stack([x[s:s+context_length] for s in starts])
    target_batch = np.stack([x[s+1:s+1+context_length] for s in starts])

    input_tensor = torch.tensor(input_batch, dtype=torch.long, device=device)
    target_tensor = torch.tensor(target_batch, dtype=torch.long, device=device)

    return input_tensor, target_tensor


tree = logging.getLogger(__name__)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the cross-entropy loss between the given logits and targets.

    Args:
        logits (torch.Tensor): The input tensor of shape (batch_size, num_classes)
            containing the unnormalized log-probabilities of each class.
        targets (torch.Tensor): The input tensor of shape (batch_size) containing
            the index of the correct class.

    Returns:
        torch.Tensor: The cross-entropy loss of shape () containing the mean loss
            across the batch.
    """
    logit_max = torch.amax(logits, dim=-1, keepdim=True)
    logits_stable = logits - logit_max
    log_sum_exp = torch.log(torch.exp(logits_stable).sum(dim=-1, keepdim=True))
    log_probs = logits_stable - log_sum_exp
    loss = -torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return loss.mean()


class CheckpointHandler:
    """Handles saving/loading checkpoints, either best or every."""
    def __init__(self,
                 path: Union[str, os.PathLike],
                 keep_best_k: int = 1):
        self.base_path = str(path)
        self.keep_best_k = keep_best_k
        self.best_list: List[Tuple[float, str]] = []  # (val_loss, filepath)

    def save(self,
             model: torch.nn.Module,
             optimizer: torch.optim.Optimizer,
             epoch: int,
             global_step: int,
             val_loss: Optional[float] = None,
             best_only: bool = False) -> None:
        """
        If best_only=True, save only when a new best val_loss and keep top-K.
        Else always save.
        """
        # Determine filename
        if best_only and val_loss is not None:
            filename = f"{os.path.splitext(self.base_path)[0]}_best_{val_loss:.4f}.ckpt"
        else:
            filename = f"{os.path.splitext(self.base_path)[0]}_epoch{epoch}_step{global_step}.ckpt"

        # Save state
        state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "val_loss": val_loss
        }
        torch.save(state, filename)
        tree.info(f"Checkpoint saved: {filename}")

        # If best_only, maintain top-K
        if best_only and val_loss is not None:
            self.best_list.append((val_loss, filename))
            self.best_list.sort(key=lambda x: x[0])
            if len(self.best_list) > self.keep_best_k:
                _, old_file = self.best_list.pop(-1)
                if os.path.isfile(old_file):
                    os.remove(old_file)
                    tree.info(f"Removed old best checkpoint: {old_file}")

    def load(self,
             model: torch.nn.Module,
             optimizer: torch.optim.Optimizer) -> Tuple[int, int]:
        if not os.path.isfile(self.base_path):
            tree.warning(f"No checkpoint found at {self.base_path}, starting fresh.")
            return 0, 0
        ckpt = torch.load(self.base_path)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        epoch = ckpt.get('epoch', 0)
        step = ckpt.get('global_step', 0)
        tree.info(f"Loaded checkpoint at epoch {epoch}, step {step}")
        return epoch, step

class Logger:
    def __init__(self, use_wandb: bool, project_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.use_wandb = use_wandb
        try:
            if use_wandb:
                assert project_name, "project_name required for W&B"
                wandb.init(project=project_name, config=config)
        except Exception as e:
            print(f"wandb disabled due to: {e}")
            self.use_wandb = False
            
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        metrics_str = ", ".join([f"{k}={v:.4f}" for k,v in metrics.items()])
        tree.info(f"Step {step}: {metrics_str}")
        if self.use_wandb:
            wandb.log(metrics, step=step)

class TrainManager:
    def __init__(self,
                 model: torch.nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 optimizer: torch.optim.Optimizer,
                 config: Dict[str, Any]):
        self.model = model
        self.model = torch.compile(self.model)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = cross_entropy
        self.scaler = torch.GradScaler(device="cuda")
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Handlers
        keep_k = config.get('keep_best_k', 1)
        self.ckpt_handler = CheckpointHandler(config['checkpoint_path'], keep_best_k=keep_k)
        self.logger = Logger(config.get('use_wandb', False), config.get('wandb_project'), config)

        # LR scheduler params
        self.a_max = config.get('lr_max', 1e-3)
        self.a_min = config.get('lr_min', 1e-5)
        self.T_w = config.get('warmup_steps', 1000)
        self.T_c = config.get('total_steps', 10000)

        # Checkpointing options
        self.save_best_only = config.get('save_best_only', False)
        self.keep_best_k = keep_k

        # Resume
        self.start_epoch, self.global_step = self.ckpt_handler.load(self.model, self.optimizer)

    def train(self) -> None:
        num_epochs = self.config.get('epochs', 1)
        val_every = self.config.get('val_every', 1)

        best_val = float('inf')
        for epoch in tqdm(range(self.start_epoch, num_epochs), desc="Epochs", unit="epoch"):
            self.model.train()
            total_loss = 0.0
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False, unit="batch"):
                inputs = batch['input_ids'].to(self.device)
                targets = batch['labels'].to(self.device)
                self.optimizer.zero_grad()

                # use mix precision
                with autocast(device_type="cuda"):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                self.scaler.scale(loss).backward()

                # Grad clipping
                if self.config.get('max_grad_norm'): 
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])

                # Update LR
                self.global_step += 1
                lr = lr_cosine_schedule(self.global_step, self.a_max, self.a_min, self.T_w, self.T_c)
                for pg in self.optimizer.param_groups:
                    pg['lr'] = lr

                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()

            avg_train = total_loss / len(self.train_loader)
            self.logger.log_metrics({'train_loss': avg_train, 'lr': lr}, step=self.global_step)

            # Validation
            if (epoch + 1) % val_every == 0:
                val_loss = self.validate()
                self.logger.log_metrics({'val_loss': val_loss}, step=self.global_step)

                # Checkpoint: either best or every
                if self.save_best_only:
                    if val_loss < best_val:
                        best_val = val_loss
                        self.ckpt_handler.save(self.model, self.optimizer, epoch+1, self.global_step, val_loss, best_only=True)
                else:
                    self.ckpt_handler.save(self.model, self.optimizer, epoch+1, self.global_step)

    def validate(self) -> float:
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", unit="batch"):
                inp = batch['input_ids'].to(self.device)
                tgt = batch['labels'].to(self.device)
                out = self.model(inp)
                total += self.criterion(out, tgt).item()
        avg = total / len(self.val_loader)
        tree.info(f"Validation Loss = {avg:.4f}")
        return avg


class MemmapTokenDataset(Dataset):
    def __init__(self, file_path, context_length):
        super().__init__()
        self.data = np.memmap(file_path, dtype=np.uint16, mode="r")
        self.context_length = context_length
        self.total_len = len(self.data) - context_length

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        assert idx < self.total_len, f"idx {idx} is out of bounds"
        assert self.data is not None, "data is None"
        
        x = self.data[idx : idx + self.context_length]
        y = self.data[idx + 1 : idx + 1 + self.context_length]

        return {
            "input_ids": torch.tensor(x, dtype=torch.long),
            "labels": torch.tensor(y, dtype=torch.long)
        }