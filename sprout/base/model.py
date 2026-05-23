import warnings

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.optim import AdamW

from sprout.base.config import Config


class MultiHeadAttention(nn.Module):
    """Causal Multi-Head Self-Attention module with Flash Attention support."""
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.dropout = config.dropout
        self.q_attn = nn.Linear(self.d_model, self.d_model)
        self.k_attn = nn.Linear(self.d_model, self.d_model)
        self.v_attn = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj.is_residual = True # Mark as residual for down scaling in weight initialization
        self.attn_dropout = nn.Dropout(self.dropout)
        self.register_buffer("mask", torch.ones(config.max_seq_len, config.max_seq_len).tril().bool()) # Causal mask for attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q = self.q_attn(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.k_attn(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.v_attn(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        try:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        except RuntimeError: # Fallback to manual attention if Flash Attention is not available
            warnings.warn("Flash Attention not available, falling back to manual attention implementation.")
            scale_factor = (C // self.num_heads) ** -0.5
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
            attn_scores = attn_scores.masked_fill(~self.mask[:T, :T], float("-inf")) # Slice mask to match sequence length and fill with -inf
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C) # Reshape back to input shape
        return self.out_proj(attn_output)

class FeedForward(nn.Module):
    """Simple Feed-Forward Network with GELU activation."""
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, 4 * config.d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(4 * config.d_model, config.d_model)
        self.linear2.is_residual = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return self.linear2(x)

class TransformerBlock(nn.Module):
    """Single Transformer block consisting of Multi-Head Attention and Feed-Forward Network."""
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class TransformerDecoder(nn.Module):
    """Transformer Decoder consisting of multiple Transformer blocks."""
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.num_layers = config.num_layers
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_embed = nn.Embedding(config.max_seq_len, config.d_model)
        self.register_buffer("positional_ids", torch.arange(config.max_seq_len).unsqueeze(0).expand(1, config.max_seq_len))
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False) # No bias because token_embed has no bias
        self.lm_head.weight = self.token_embed.weight # Weight tying
        self.config = config
        self.apply(self._init_weights)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.size()
        token_embeddings = self.token_embed(input_ids)
        positional_ids = self.positional_ids[:, :T]
        positional_embeddings = self.positional_embed(positional_ids)
        x = token_embeddings + positional_embeddings
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

    def num_parameters(self) -> int:
        """Returns the total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def configure_optimizers(self, config: Config):
        """Configures the AdamW optimizer with the specified hyperparameters from the config."""
        return AdamW(
            self.parameters(),
            lr=config.max_learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps,
            fused=True
        )
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        if hasattr(module, "is_residual") and module.is_residual:
            nn.init.normal_(module.weight, mean=0.0, std=0.02 / (2 * self.num_layers)**0.5)