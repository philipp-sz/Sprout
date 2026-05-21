import warnings

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.optim import AdamW, Muon

from sprout.modern.config import Config


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) module."""
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.head_dim = config.d_model // config.num_heads
        self.max_seq_len = config.max_seq_len
        self.base_freq = config.rope_base_freq

        inv_freq = 1.0 / (self.base_freq ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.sin_cached = None
        self.cos_cached = None
        self._set_cos_sin_cache(self.max_seq_len) # Precompute cos and sin values for the maximum sequence length

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Applies rotary positional embeddings to the input tensor."""
        if seq_len > self.max_seq_len: # Recompute cache if sequence length exceeds current max
            self.max_seq_len = seq_len
            self._set_cos_sin_cache(seq_len)

        return self._apply_rotary_pos_emb(q, k, seq_len)

    def _set_cos_sin_cache(self, seq_len: int) -> None:
        """Precomputes and caches the cosine and sine values for the given sequence length."""
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1) # Duplicate frequencies to work with head_dim

        # Add singletons for broadcasting during attention computation
        self.cos_cached = emb.cos().unsqueeze(0).unsqueeze(2) # Shape (1, seq_len, 1, head_dim)
        self.sin_cached = emb.sin().unsqueeze(0).unsqueeze(2) # Shape (1, seq_len, 1, head_dim)

    def _apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Applies the precomputed rotary positional embeddings to the query and key tensors."""
        # Slice the cached cos and sin values to match the current sequence length
        cos = self.cos_cached[:, :seq_len, :, :].to(device=q.device, dtype=q.dtype)
        sin = self.sin_cached[:, :seq_len, :, :].to(device=q.device, dtype=q.dtype)

        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)

        return q_rotated, k_rotated

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input to efficiently compute the rotation."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

class MultiHeadAttention(nn.Module):
    """Causal Multi-Head Self-Attention module with Flash Attention support."""
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_size = config.d_model // config.num_heads
        self.dropout = config.dropout

        self.rope = RotaryEmbedding(config)
        self.q_attn = nn.Linear(config.d_model, config.num_heads * self.head_size, bias=False)
        self.k_attn = nn.Linear(config.d_model, config.num_kv_heads * self.head_size, bias=False)
        self.v_attn = nn.Linear(config.d_model, config.num_kv_heads * self.head_size, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.head_size, dtype=torch.bfloat16)
        self.k_norm = nn.RMSNorm(self.head_size, dtype=torch.bfloat16)
        self.out_proj.is_residual = True # Mark as residual for down scaling in weight initialization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.register_buffer("mask", torch.ones(config.max_seq_len, config.max_seq_len).tril().bool()) # Causal mask for attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q = self.q_attn(x).view(B, T, self.num_heads, self.head_size)
        k = self.k_attn(x).view(B, T, self.num_kv_heads, self.head_size)
        v = self.v_attn(x).view(B, T, self.num_kv_heads, self.head_size)
        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = self.rope(q, k, seq_len=T) 
        q = q.transpose(1, 2) 
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        try:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True, dropout_p=self.dropout if self.training else 0.0)
        except RuntimeError: # Fallback to manual attention if Flash Attention is not available
            warnings.warn("Flash Attention not available, falling back to manual attention implementation.")
            # Repeat keys and values to match queries for GQA
            groups = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)

            scale_factor = (C // self.num_heads) ** -0.5
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
            attn_scores = attn_scores.masked_fill(~self.mask[:T, :T], float("-inf")) # Slice mask to match sequence length and fill with -inf
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C) # Reshape back to input shape
        return self.out_proj(attn_output)

class FeedForward(nn.Module):
    """Simple Feed-Forward Network with ReLU^2 activation."""
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, int(config.mlp_expansion*config.d_model), bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(int(config.mlp_expansion*config.d_model), config.d_model, bias=False)
        self.linear2.is_residual = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.relu(x).square()
        x = self.dropout(x)
        return self.linear2(x)

class TransformerBlock(nn.Module):
    """Single Transformer block consisting of Multi-Head Attention and Feed-Forward Network."""
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.norm1 = nn.RMSNorm(config.d_model)
        self.norm2 = nn.RMSNorm(config.d_model)

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
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False) # No bias because token_embed has no bias
        self.lm_head.weight = self.token_embed.weight # Weight tying
        self.config = config
        self.apply(self._init_weights)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embed(input_ids)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

    def num_parameters(self) -> int:
        """Returns the total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def configure_optimizers(self, config: Config):
        """Configures the AdamW optimizer with the specified hyperparameters from the config."""
        adamw_params, muon_params = self._identify_optimizer_params()
        opt_adam = AdamW(
            adamw_params,
            lr=config.max_learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps,
            fused=True
        )
        opt_muon = Muon(
            muon_params,
            lr=config.max_learning_rate,
            weight_decay=config.weight_decay,
            momentum=config.muon_momentum,
            adjust_lr_fn="match_rms_adamw",
        )
        return opt_adam, opt_muon
    
    def _identify_optimizer_params(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        """Identifies parameter groups for muon and adamw optimizers."""
        muon_params = []
        adamw_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # Only apply Muon to 2D weight matrices and not to embeddings
            if param.ndim == 2 and "token_embed" not in name and "lm_head" not in name:
                muon_params.append(param)
            else:
                adamw_params.append(param)
        return adamw_params, muon_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        if hasattr(module, "is_residual") and module.is_residual:
            nn.init.normal_(module.weight, mean=0.0, std=0.02 / (2 * self.num_layers)**0.5)