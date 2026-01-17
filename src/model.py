"""
Chess Transformer Model for the Chess Challenge.

Modifications:
- RoPE positional encoding controlled by config.use_rope (default: True)
- Optional one-hot embeddings controlled by config.one_hot_embeds (default: False)
- GPU/torch.compile-friendly attention via torch.nn.functional.scaled_dot_product_attention (SDPA)

Key components:
- ChessConfig: Configuration class for model hyperparameters
- ChessForCausalLM: The main model class for next-move prediction
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class ChessConfig(PretrainedConfig):
    """
    Configuration class for the Chess Transformer model.

    New attributes:
        use_rope: If True, use Rotary Positional Embeddings (RoPE) instead of learned absolute positions.
        rope_theta: Base (theta) for RoPE frequencies (default: 10000.0).
        one_hot_embeds: If True, compute token embeddings via one-hot -> matmul with embedding matrix.
                        (More expensive; intended for experiments.)
    """

    model_type = "chess_transformer"

    def __init__(
        self,
        vocab_size: int = 1200,
        n_embd: int = 128,
        n_layer: int = 6,
        n_head: int = 4,
        n_ctx: int = 256,
        n_inner: Optional[int] = None,
        dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        tie_weights: bool = True,
        # NEW:
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        one_hot_embeds: bool = False,
        # Tokens:
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        self.vocab_size = vocab_size

        self.one_hot_embeds = bool(one_hot_embeds)
        d_model = self.vocab_size if self.one_hot_embeds else n_embd

        self.n_embd = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_ctx = n_ctx
        self.n_inner = n_inner if n_inner is not None else 3 * n_embd  # keep your budget choice
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon

        self.tie_weights = bool(tie_weights) and (not self.one_hot_embeds)
        self.tie_word_embeddings = bool(self.tie_weights)


        self.use_rope = bool(use_rope)
        self.rope_theta = float(rope_theta)
        self.one_hot_embeds = bool(one_hot_embeds)

        # Inform HF base class about tying behavior
        self.tie_word_embeddings = bool(tie_weights)

        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head}). "
                f"(If one_hot_embeds=True, n_embd=vocab_size={self.vocab_size})"
            )

        head_dim = self.n_embd // self.n_head
        if self.use_rope and (head_dim % 2 != 0):
            raise ValueError(
                f"RoPE requires even head_dim, got head_dim={head_dim}. "
                f"Choose n_head such that (n_embd/n_head) is even."
            )



class OneHotEmbedding(nn.Module):
    """
    True one-hot embedding:
    token i -> e_i in R^V (V = vocab_size)

    - No parameters
    - No embedding matrix
    - Returns a dense (B, L, V) tensor (this is inherently expensive)
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = int(vocab_size)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        # Pick a dtype that matches autocast (saves memory in bf16/fp16)
        if torch.is_autocast_enabled():
            if input_ids.is_cuda:
                dtype = torch.get_autocast_gpu_dtype()
            else:
                dtype = torch.get_autocast_cpu_dtype()
        else:
            dtype = torch.float32

        # Allocate the dense one-hot tensor directly in compute dtype
        # Shape: (B, L, V)
        out = torch.zeros(
            (*input_ids.shape, self.vocab_size),
            device=input_ids.device,
            dtype=dtype,
        )
        out.scatter_(-1, input_ids.unsqueeze(-1), 1.0)
        return out


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Simpler and faster than LayerNorm - used in LLaMA, Mistral, etc.
    Does not center (no mean subtraction), only scales by RMS.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        # Compute RMS: sqrt(mean(x^2))
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) with precomputed sin/cos cache.

    Cache is created up to max_position_embeddings and stored as buffers (not in state_dict).
    Applies RoPE to Q and K in (B, H, L, D) format.
    """

    def __init__(self, dim: int, max_position_embeddings: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE requires an even dim, got dim={dim}")
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings

        # inv_freq: (dim/2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin: (max_pos, dim/2)
        t = torch.arange(max_position_embeddings, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    @staticmethod
    def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: (B, H, L, D)
        # cos/sin: broadcastable to (B or 1, 1, L, D/2)
        x1 = x[..., ::2]  # (B,H,L,D/2)
        x2 = x[..., 1::2]  # (B,H,L,D/2)

        # Apply rotation
        # [x1; x2] -> [x1*cos - x2*sin ; x1*sin + x2*cos]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos

        # Interleave back to (B,H,L,D)
        return torch.stack((y1, y2), dim=-1).flatten(-2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # q,k: (B, H, L, D)
        B, H, L, D = q.shape

        if position_ids is None:
            # Fast path: positions [0..L-1], same for whole batch
            if L > self.cos_cached.size(0):
                raise ValueError(
                    f"Sequence length {L} exceeds RoPE cache size {self.cos_cached.size(0)}. "
                    f"Increase n_ctx/max_position_embeddings."
                )
            cos = self.cos_cached[:L].to(dtype=q.dtype).unsqueeze(0).unsqueeze(0)  # (1,1,L,D/2)
            sin = self.sin_cached[:L].to(dtype=q.dtype).unsqueeze(0).unsqueeze(0)
        else:
            # position_ids: (B, L) (or (L,))
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0).expand(B, -1)

            flat = position_ids.reshape(-1)  # (B*L,)
            cos = (
                self.cos_cached.index_select(0, flat)
                .reshape(B, L, -1)
                .to(dtype=q.dtype)
                .unsqueeze(1)  # (B,1,L,D/2)
            )
            sin = (
                self.sin_cached.index_select(0, flat)
                .reshape(B, L, -1)
                .to(dtype=q.dtype)
                .unsqueeze(1)  # (B,1,L,D/2)
            )

        q = self._apply_rotary(q, cos, sin)
        k = self._apply_rotary(k, cos, sin)
        return q, k


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention using SDPA, with correct padding masking.

    - If attention_mask is provided, we build a boolean "keep mask" that combines:
        * causal mask (lower triangular)
        * key padding mask
      and call SDPA with is_causal=False (mask already contains causal).
    - If attention_mask is None, we call SDPA with is_causal=True (fast path).
    """

    def __init__(self, config: ChessConfig):
        super().__init__()

        assert config.n_embd % config.n_head == 0, (
            f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
        )

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout_p = float(config.dropout)

        self.use_rope = bool(getattr(config, "use_rope", False))
        if self.use_rope:
            if self.head_dim % 2 != 0:
                raise ValueError(f"RoPE requires even head_dim, got {self.head_dim}")
            self.rope = RotaryEmbedding(
                dim=self.head_dim,
                max_position_embeddings=config.n_ctx,
                base=float(getattr(config, "rope_theta", 10000.0)),
            )
        else:
            self.rope = None

        # Causal keep-mask buffer: True means "allowed"
        self.register_buffer(
            "causal_keep",
            torch.tril(torch.ones(config.n_ctx, config.n_ctx, dtype=torch.bool)).view(
                1, 1, config.n_ctx, config.n_ctx
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, _ = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.reshape(B, L, self.n_head, self.head_dim).transpose(1, 2)  # (B,H,L,D)
        k = k.reshape(B, L, self.n_head, self.head_dim).transpose(1, 2)
        v = v.reshape(B, L, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q, k = self.rope(q, k, position_ids=position_ids)

        dropout_p = self.attn_dropout_p if self.training else 0.0

        # Correct masking (equivalent to the old code):
        # - Old code: causal mask + key padding mask applied to attention scores.
        # - Here: build a boolean keep-mask (True=keep, False=masked) for SDPA.
        if attention_mask is None:
            attn_mask = None
            is_causal = True
        else:
            # key_keep: (B,1,1,L) True for real tokens, False for pads
            key_keep = attention_mask[:, None, None, :].to(dtype=torch.bool)
            # causal_keep: (1,1,L,L)
            causal_keep = self.causal_keep[:, :, :L, :L]
            # combined: (B,1,L,L) via broadcast
            attn_mask = causal_keep & key_keep
            is_causal = False  # mask already contains causal

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )  # (B,H,L,D)

        attn_output = attn_output.transpose(1, 2).reshape(B, L, self.n_embd)
        return self.c_proj(attn_output)



class FeedForward(nn.Module):
    """
    Feed-forward network (MLP) module.

    Standard two-layer MLP with GELU activation.
    """

    def __init__(self, config: ChessConfig):
        super().__init__()

        self.c_fc = nn.Linear(config.n_embd, config.n_inner)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    A single transformer block with attention and feed-forward layers.

    Uses pre-normalization (LayerNorm before attention/FFN) for better
    training stability.
    """

    def __init__(self, config: ChessConfig):
        super().__init__()

        self.ln_1 = RMSNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask, position_ids=position_ids)
        x = x + self.mlp(self.ln_2(x))
        return x


class ChessForCausalLM(PreTrainedModel):
    """
    Chess Transformer for Causal Language Modeling (next-move prediction).

    RoPE:
      - If config.use_rope=True (default), no learned positional embeddings are used.
      - RoPE is applied inside attention on Q and K.

    One-hot embeddings:
      - If config.one_hot_embeds=True, input embeddings are computed as:
          one_hot(input_ids) @ wte.weight
        This is heavier than nn.Embedding lookup, but matches the requested behavior.
    """

    config_class = ChessConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    keys_to_ignore_on_load_missing = ["lm_head.weight"]

    def __init__(self, config: ChessConfig):
        super().__init__(config)

        if config.one_hot_embeds:
            self.wte = OneHotEmbedding(config.vocab_size)
        else:
            self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Positional embeddings only if not using RoPE
        self.wpe = None if getattr(config, "use_rope", False) else nn.Embedding(config.n_ctx, config.n_embd)

        self.drop = nn.Dropout(config.dropout)

        self.h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])

        self.ln_f = RMSNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.tie_weights:
            self._tied_weights_keys = ["lm_head.weight"]

        self.post_init()

        if config.tie_weights and (not config.one_hot_embeds):
            self.tie_weights()

    def get_input_embeddings(self) -> nn.Module:
        return self.wte

    def set_input_embeddings(self, new_embeddings: nn.Module):
        self.wte = new_embeddings
        if getattr(self.config, "tie_weights", False):
            self.tie_weights()

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module):
        self.lm_head = new_embeddings

    def tie_weights(self):
        if getattr(self.config, "one_hot_embeds", False):
            return
        if getattr(self.config, "tie_weights", False) or getattr(self.config, "tie_word_embeddings", False):
            self._tie_or_clone_weights(self.lm_head, self.wte)


    def _init_weights(self, module: nn.Module):
        """Initialize weights following GPT-2 style."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        B, L = input_ids.size()
        device = input_ids.device

        use_rope = bool(getattr(self.config, "use_rope", False))
        one_hot_embeds = bool(getattr(self.config, "one_hot_embeds", False))

        # Only build position_ids when needed for learned absolute positions.
        # For RoPE, position_ids can be None (fast path), unless caller provides custom position_ids.
        if (position_ids is None) and (not use_rope):
            position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

        # Token embeddings
        if one_hot_embeds:
            token_embeds = self.wte(input_ids)
            hidden_states = token_embeds
        else:
            token_embeds = self.wte(input_ids)

        hidden_states = token_embeds

        # Absolute learned positions only if RoPE disabled
        if not use_rope:
            if self.wpe is None:
                raise RuntimeError("wpe is None but use_rope is False (inconsistent init).")
            pos_embeds = self.wpe(position_ids)
            hidden_states = hidden_states + pos_embeds

        # Optional: zero out padded positions early (cheap)
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)

        hidden_states = self.drop(hidden_states)

        for block in self.h:
            hidden_states = block(hidden_states, attention_mask=attention_mask, position_ids=position_ids)

        hidden_states = self.ln_f(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    @torch.no_grad()
    def generate_move(
        self,
        input_ids: torch.LongTensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> int:
        self.eval()

        outputs = self(input_ids)
        logits = outputs.logits[:, -1, :] / temperature

        if top_k is not None:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token.item()


# Register the model with Auto classes for easy loading
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("chess_transformer", ChessConfig)
AutoModelForCausalLM.register(ChessConfig, ChessForCausalLM)
