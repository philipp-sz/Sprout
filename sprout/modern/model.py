import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.optim import AdamW, Muon

from sprout.modern.config import Config


class KVCache:
    """Key-Value Cache for storing past keys and values during autoregressive decoding."""
    def __init__(self, config: Config, batch_size: int):
        self.max_seq_len = config.max_seq_len
        self.cur_step = 0  # Tracks where to insert the next token
        
        # Pre-allocate with max_seq_len
        self.k_cache = torch.zeros(config.num_layers, batch_size, config.num_kv_heads, config.max_seq_len, config.head_dim, device=config.device, dtype=config.dtype)
        self.v_cache = torch.zeros(config.num_layers, batch_size, config.num_kv_heads, config.max_seq_len, config.head_dim, device=config.device, dtype=config.dtype)

    def update(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Writes the new keys and values of one layer and returns all cached keys and values of that layer."""
        end_idx = self.cur_step + k_new.size(-2)
        if end_idx > self.max_seq_len:
            raise ValueError(f"KV cache overflow: {end_idx} tokens exceed max_seq_len={self.max_seq_len}. Re-prefill a shorter window.")

        self.k_cache[layer_idx, :, :, self.cur_step:end_idx, :] = k_new
        self.v_cache[layer_idx, :, :, self.cur_step:end_idx, :] = v_new
        return self.k_cache[layer_idx, :, :, :end_idx, :], self.v_cache[layer_idx, :, :, :end_idx, :]

    def reset(self) -> None:
        """Empties the cache; old entries are overwritten by the next update."""
        self.cur_step = 0

    def advance(self, seq_len: int) -> None:
        """Moves the insert position forward once all layers have processed the new tokens."""
        self.cur_step += seq_len

    @property
    def current_step(self) -> int:
        """Returns the current decoding step, which indicates how many tokens are currently in the cache."""
        return self.cur_step

class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) module."""
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.base_freq = config.rope_base_freq

        inv_freq = 1.0 / (self.base_freq ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.sin_cached = None
        self.cos_cached = None
        self._set_cos_sin_cache(self.max_seq_len) # Precompute cos and sin values for the maximum sequence length

    def forward(self, q: Tensor, k: Tensor, seq_len: int, pos_offset: int | None = None) -> Tensor:
        """Applies rotary positional embeddings to the input tensor."""
        if pos_offset + seq_len > self.max_seq_len: # Extend the cache if positions go beyond the precomputed ones
            self._set_cos_sin_cache(pos_offset + seq_len)

        return self._apply_rotary_pos_emb(q, k, seq_len, pos_offset)

    def _set_cos_sin_cache(self, seq_len: int) -> None:
        """Precomputes and caches the cosine and sine values for the given sequence length."""
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1) # Duplicate frequencies to work with head_dim

        # Add singletons for broadcasting during attention computation
        self.cos_cached = emb.cos().unsqueeze(0).unsqueeze(2) # Shape (1, seq_len, 1, head_dim)
        self.sin_cached = emb.sin().unsqueeze(0).unsqueeze(2) # Shape (1, seq_len, 1, head_dim)

    def _apply_rotary_pos_emb(self, q: Tensor, k: Tensor, seq_len: int, pos_offset: int | None = None) -> Tensor:
        """Applies the precomputed rotary positional embeddings to the query and key tensors."""
        # Slice the cached cos and sin values to match the current sequence length
        cos = self.cos_cached[:, pos_offset:pos_offset + seq_len, :, :].to(device=q.device, dtype=q.dtype)
        sin = self.sin_cached[:, pos_offset:pos_offset + seq_len, :, :].to(device=q.device, dtype=q.dtype)

        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)

        return q_rotated, k_rotated

    def _rotate_half(self, x: Tensor) -> Tensor:
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

    def forward(self, x: Tensor, kv_cache: KVCache | None = None, layer_idx: int | None = None) -> Tensor:
        B, T, C = x.size()
        q = self.q_attn(x).view(B, T, self.num_heads, self.head_size)
        k = self.k_attn(x).view(B, T, self.num_kv_heads, self.head_size)
        v = self.v_attn(x).view(B, T, self.num_kv_heads, self.head_size)
        q = self.q_norm(q)
        k = self.k_norm(k)

        pos_offset = kv_cache.current_step if kv_cache is not None else 0
        q, k = self.rope(q, k, seq_len=T, pos_offset=pos_offset) 
        q = q.transpose(1, 2) 
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx=layer_idx, k_new=k, v_new=v)

        attn_mask = None
        is_causal = False
        if T > 1:
            T_kv = k.size(-2)

            if T_kv == T:
                is_causal = True
            else:
                q_idx = torch.arange(T, device=q.device).unsqueeze(1)
                kv_idx = torch.arange(T_kv, device=k.device).unsqueeze(0)
                mask_bool = kv_idx <= (T_kv - T + q_idx)

                mask_value = -10000.0 if q.dtype in [torch.float16, torch.bfloat16] else float("-inf")
                attn_mask = torch.zeros(T, T_kv, device=q.device, dtype=q.dtype)
                attn_mask = attn_mask.masked_fill(~mask_bool, mask_value)

        attn_output = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attn_mask,
                is_causal=is_causal,
                enable_gqa=True, 
                dropout_p=self.dropout if self.training else 0.0
            )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C) 
        return self.out_proj(attn_output)
    
    def _calc_manual_attention(self, q: Tensor, k: Tensor, v: Tensor, C: int, T: int) -> Tensor:
        """Manual implementation of scaled dot-product attention with causal masking."""
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
        return attn_output

class FeedForward(nn.Module):
    """Simple Feed-Forward Network with ReLU^2 activation."""
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, int(config.mlp_expansion*config.d_model), bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(int(config.mlp_expansion*config.d_model), config.d_model, bias=False)
        self.linear2.is_residual = True

    def forward(self, x: Tensor) -> Tensor:
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

    def forward(self, x: Tensor, kv_cache: KVCache | None = None, layer_idx: int | None = None) -> Tensor:
        x = x + self.attn(self.norm1(x), kv_cache=kv_cache, layer_idx=layer_idx)
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

    def __call__(self, input_ids: Tensor, kv_cache: KVCache | None = None) -> Tensor:
        return self.forward(input_ids, kv_cache)

    def forward(self, input_ids: Tensor, kv_cache: KVCache | None = None) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        x = self.token_embed(input_ids)

        for i, layer in enumerate(self.layers):
            layer_idx = i if kv_cache is not None else None
            x = layer(x, kv_cache=kv_cache, layer_idx=layer_idx)
        if kv_cache is not None: # Advance once per forward pass, after every layer has written its keys and values
            kv_cache.advance(input_ids.size(1))

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