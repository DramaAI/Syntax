from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import math


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class CasualSelfAttention(nn.Module):
    """
    Casual Self-Attention module used in GPT (Generative Pre-trained Transformer) models.

    Args:
        config (GPTConfig): An instance of the GPT configuration containing model hyperparameters. Default is None.
        
    Attributes:
        c_attn: Linear layer for computing queries, keys, and values.
        c_proj: Linear layer for projecting the output.
        attn_dropout: Dropout applied to the attention scores.
        n_embd: Dimension of the model's embeddings.
        n_head: Number of attention heads.
        dropout: Dropout probability for attention.

    Methods:
        forward(input): Forward pass of the Casual Self-Attention module.

    Example:
        # Create a Casual Self-Attention module using a specific configuration
        config = GPTConfig(d_model=512, n_head=8, d_ffn=2048, dropout=0.1)
        self_attn = CasualSelfAttention(config)
    """

    def __init__(self, config: Optional[GPTConfig]=None) -> None:
        super().__init__()

        # Linear layers for queries, keys, and values
        self.c_attn = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

        # Dropout for attention scores
        self.attn_dropout = nn.Dropout(config.dropout)

        # Model hyperparameters
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.dropout = config.dropout

        # Attention bias matrix
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                                          .view(1, 1, config.block_size, config.block_size))

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass of the Casual Self-Attention module.

        Args:
            input (Tensor): Input tensor to the Casual Self-Attention module.

        Returns:
            Tensor: Output tensor after applying self-attention mechanism.

        The forward pass of the Casual Self-Attention module involves linear transformations for queries, keys, and values, followed by self-attention computation, masking, and linear projection.
        """
        B, T, C = input.size()

        # Split the output of the linear transformation into q, k, and v
        q, k, v = self.c_attn(input).split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Calculate attention scores
        attn = q @ k.transpose(-1, -2) * (1.0 / math.sqrt(k.size()))

        # Apply masking to ensure causality
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        # Apply softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Compute the weighted sum of values
        attn = attn @ v

        # Reshape and concatenate the attention heads
        y = attn.transpose(1, 2).contiguous().view(B, T, C)
        return y


class MLP(nn.Module):

    def __init__(self, config : GPTConfig=None) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.drouput = nn.Dropout(config.dropout)

    def forward(self, input : Tensor):
        return self.drouput(
            self.c_proj(
                self.gelu(
                    self.c_fc(input)
                )
            )
        )
    
class Block(nn.Module):
    """
    Transformer Block for GPT (Generative Pre-trained Transformer) models.

    Args:
        config (GPTConfig): An instance of the GPT configuration containing model hyperparameters.

    Attributes:
        ln_1: Layer normalization module after the first sub-layer.
        attn: CasualSelfAttention module for self-attention mechanism.
        ln_2: Layer normalization module after the second sub-layer.
        mlp: MLP (Multi-Layer Perceptron) module for feed-forward operations.

    Methods:
        forward(input): Forward pass of the Transformer block.

    Example:
        # Create a Transformer block for a GPT model using a specific configuration
        config = GPTConfig(d_model=512, n_head=8, d_ffn=2048, dropout=0.1)
        transformer_block = Block(config)
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)  # Replace with actual layer normalization module
        self.attn = CasualSelfAttention(config)  # Assuming CasualSelfAttention is a defined module
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)  # Replace with actual layer normalization module
        self.mlp = MLP(config)  # Assuming MLP is a defined module

    def forward(self, input : Tensor) -> Tensor:
        """
        Forward pass of the Transformer block.

        Args:
            input (Tensor): Input tensor to the Transformer block.

        Returns:
            Tensor: Output tensor after passing through the block.

        The forward pass of the Transformer block involves applying layer normalization, self-attention, and a feed-forward neural network in sequence.
        """
        input = self.attn(self.ln_1(input))
        input = self.mlp(self.ln_2(input))
        return input

class LayerNorm(nn.Module):
    """
    Layer normalization with optional bias.

    This class implements layer normalization, similar to PyTorch's native layer normalization, but with an option to include bias or not.

    Args:
        ndim (int): The number of dimensions of the input tensor. Typically corresponds to the number of features.
        bias (bool): If True, bias terms are included. If False, no bias terms are applied.

    Attributes:
        weight (nn.Parameter): Learnable weight parameter for scaling the normalized values.
        bias (nn.Parameter or None): Learnable bias parameter for adding an offset to the normalized values. If bias is not used (bias=False), this attribute is set to None.

    Methods:
        forward(input): Forward pass of the layer normalization.

    Example:
        # Create a layer normalization layer with bias
        layer_norm_with_bias = LayerNorm(64, bias=True)

        # Create a layer normalization layer without bias
        layer_norm_without_bias = LayerNorm(64, bias=False)
    """

    def __init__(self, ndim : int, bias : bool=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        """
        Forward pass of the layer normalization.

        Args:
            input (torch.Tensor): Input tensor to be normalized.

        Returns:
            torch.Tensor: Normalized tensor after applying layer normalization.

        The forward pass of the layer normalization applies the normalization transformation to the input tensor.
        """
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class SyntaxGPT(nn.Module):

    def __init__(self, config : GPTConfig) -> None:
        super().__init__()
        self.config : GPTConfig = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([
                Block(config) for _ in range(config.n_layer)
            ]),
            ln_f=LayerNorm(config.n_embd, config.bias)
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._xavier_initialization)

    
    def _xavier_initialization(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias) 
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input : Tensor):
        device = input.device
        _, t = input.size()
        assert t <= self.config.block_size
        pos = torch.arrange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(input)
        pos_emb = self.transformer.wpe(pos)

        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x[:, [-1], :])
        
        return logits
    
