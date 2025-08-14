Self-Coded Transformer Language Model
This repository contains a from-scratch implementation of a modern Transformer-based Language Model using PyTorch. The goal is to build and train a decoder-only model with contemporary architectural choices.

Model Architecture
The model is a decoder-only transformer that follows a structure similar to models like Llama. The key features are:

Pre-Normalization: Uses RMSNorm before the self-attention and feed-forward blocks for improved training stability.

Rotary Positional Embeddings (RoPE): Injects positional information by rotating the query and key vectors instead of using traditional absolute or sinusoidal embeddings.

SwiGLU Activation: The feed-forward network uses a SwiGLU (Swish-Gated Linear Unit) activation function for better performance compared to standard ReLU.

No Biases: The Linear layers are implemented without bias terms.

Code Contents
The codebase includes all the necessary components for building, training, and running the language model.

Core Modules: Custom, self-contained implementations for:

Linear, Embedding, RMSNorm, RoPE, SwiGLU

MultiheadSelfAttention and TransformerBlock

TransformerLM: The final model that stacks the blocks.

Training Infrastructure:

TrainManager: A high-level trainer class that handles the training loop, validation, mixed-precision training (autocast), and gradient scaling.

AdamW: A from-scratch implementation of the AdamW optimizer.

lr_cosine_schedule: A learning rate scheduler with linear warmup and cosine decay.

CheckpointHandler: A utility for saving and loading model checkpoints.

Logger: A simple logger with optional Weights & Biases (wandb) integration.

Data Handling:

MemmapTokenDataset: A PyTorch Dataset that uses numpy.memmap to efficiently handle large tokenized datasets that don't fit into RAM.

Inference:

A generate method is included in the TransformerLM class, supporting sampling with temperature, top-k, and top-p.

All components, from the attention mechanism to the AdamW optimizer and training loop.
