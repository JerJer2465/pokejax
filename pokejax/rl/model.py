"""
Flax port of PokeTransformer — Actor-Critic Transformer for Pokemon battles.

Architecture:
  - Pre-LN Transformer: d_model=256, n_heads=8, n_layers=4, d_ff=768
  - TokenProjection: (embed_dim=384 + float_dim=394) → d_model via 2-layer MLP+LN
  - Embedding dims: species=96, move=48×5=240 (4 moves+last), ability=24, item=24 → 384
  - Positional embeddings: 15 learned slots
  - Actor token=13, Critic token=14 with -inf bias on [13,14] and [14,13]
  - C51 distributional value head: 51 atoms, v_min=-1.5, v_max=1.5
  - Policy head: linear → masked log-softmax

Usage (JAX):
    key = jax.random.PRNGKey(0)
    model = PokeTransformer()
    params = model.init(key, int_ids, float_feats, legal_mask)
    log_probs, value_probs, value = model.apply(params, int_ids, float_feats, legal_mask)
"""

from __future__ import annotations
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from pokejax.rl.obs_builder import FLOAT_DIM_PER_POKEMON, N_TOKENS, N_ACTIONS

# ---------------------------------------------------------------------------
# Hyperparameters — scaled up for more capacity (~3.5M params)
# ---------------------------------------------------------------------------

D_MODEL          = 256
N_HEADS          = 8
N_LAYERS         = 4
D_FF             = 768
DROPOUT          = 0.0          # disabled during JIT rollout; set per-call if needed

# Embedding dims — scaled up proportionally
SPECIES_EMBED_DIM  = 96
MOVE_EMBED_DIM     = 48     # per move (4 moves + 1 last_used)
ABILITY_EMBED_DIM  = 24
ITEM_EMBED_DIM     = 24
EMBED_DIM          = SPECIES_EMBED_DIM + 4 * MOVE_EMBED_DIM + ABILITY_EMBED_DIM + ITEM_EMBED_DIM + MOVE_EMBED_DIM
# = 96 + 192 + 24 + 24 + 48 = 384

# C51 parameters
N_ATOMS  = 51
V_MIN    = -1.5
V_MAX    =  1.5

ACTOR_IDX  = 13
CRITIC_IDX = 14

# Vocab sizes (must exceed max ID in tables; gen4 has 493 species in dex
# but Showdown uses 1515 entries including formes; move IDs go up to ~954)
N_SPECIES   = 1600
N_MOVES     = 1000
N_ABILITIES = 400
N_ITEMS     = 600

# Additive attention bias: -inf at Actor↔Critic pairs, 0 elsewhere
# Shape (1, 1, N_TOKENS, N_TOKENS) for broadcasting over (batch, heads)
def _build_attn_bias() -> jnp.ndarray:
    """Build the actor-critic attention mask once at module init."""
    bias = jnp.zeros((1, 1, N_TOKENS, N_TOKENS), dtype=jnp.float32)
    bias = bias.at[0, 0, ACTOR_IDX, CRITIC_IDX].set(-1e9)
    bias = bias.at[0, 0, CRITIC_IDX, ACTOR_IDX].set(-1e9)
    return bias


# ---------------------------------------------------------------------------
# Transformer building blocks (Flax linen)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-LN Transformer block.

    x: (batch, n_tokens, d_model)
    attn_bias: (1, 1, n_tokens, n_tokens) additive float bias
    """
    d_model: int = D_MODEL
    n_heads: int = N_HEADS
    d_ff:    int = D_FF

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        attn_bias: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        head_dim = self.d_model // self.n_heads

        # Pre-LN self-attention
        h = nn.LayerNorm()(x)
        B, T, D = h.shape
        qkv = nn.Dense(3 * self.d_model, use_bias=False)(h)  # (B, T, 3D)
        qkv = qkv.reshape(B, T, 3, self.n_heads, head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # each (B, T, H, hd)
        q = q.transpose(0, 2, 1, 3)  # (B, H, T, hd)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scale = head_dim ** -0.5
        attn_w = (q @ k.transpose(0, 1, 3, 2)) * scale  # (B, H, T, T)
        attn_w = attn_w + attn_bias                      # broadcast over B, H
        attn_w = jax.nn.softmax(attn_w, axis=-1)
        attn_out = attn_w @ v                            # (B, H, T, hd)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, D)

        attn_out = nn.Dense(self.d_model)(attn_out)
        x = x + attn_out

        # Pre-LN FFN
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.d_ff)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.d_model)(h)
        x = x + h
        return x


# ---------------------------------------------------------------------------
# Full PokeTransformer
# ---------------------------------------------------------------------------

class PokeTransformer(nn.Module):
    """
    Full Actor-Critic Transformer for Pokemon battle decisions.

    Call with:
      log_probs, value_probs, value = model.apply(
          params, int_ids, float_feats, legal_mask
      )
    """
    d_model:   int   = D_MODEL
    n_heads:   int   = N_HEADS
    n_layers:  int   = N_LAYERS
    d_ff:      int   = D_FF
    n_actions: int   = N_ACTIONS
    n_atoms:   int   = N_ATOMS
    v_min:     float = V_MIN
    v_max:     float = V_MAX

    @nn.compact
    def __call__(
        self,
        int_ids:    jnp.ndarray,   # (B, 15, 8)      int
        float_feats: jnp.ndarray,  # (B, 15, 394)    float32
        legal_mask: jnp.ndarray,   # (B, 10)         float32
        deterministic: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Returns:
          log_probs  : (B, n_actions)
          value_probs: (B, n_atoms)
          value      : (B,)
        """
        B = int_ids.shape[0]

        # --- Embeddings ---
        species = nn.Embed(N_SPECIES, SPECIES_EMBED_DIM)(int_ids[..., 0])  # (B, 15, 96)
        m0 = nn.Embed(N_MOVES, MOVE_EMBED_DIM)(int_ids[..., 1])
        m1 = nn.Embed(N_MOVES, MOVE_EMBED_DIM)(int_ids[..., 2])
        m2 = nn.Embed(N_MOVES, MOVE_EMBED_DIM)(int_ids[..., 3])
        m3 = nn.Embed(N_MOVES, MOVE_EMBED_DIM)(int_ids[..., 4])
        ability = nn.Embed(N_ABILITIES, ABILITY_EMBED_DIM)(int_ids[..., 5])
        item    = nn.Embed(N_ITEMS,     ITEM_EMBED_DIM)(int_ids[..., 6])
        last_m  = nn.Embed(N_MOVES, MOVE_EMBED_DIM)(int_ids[..., 7])

        embed_out = jnp.concatenate(
            [species, m0, m1, m2, m3, ability, item, last_m], axis=-1
        )  # (B, 15, EMBED_DIM)

        # --- Token projection: (embed_dim + float_dim) → d_model ---
        token_in = jnp.concatenate([embed_out, float_feats], axis=-1)  # (B, 15, EMBED_DIM+394)
        BT = B * N_TOKENS
        tok_flat = token_in.reshape(BT, -1)                            # (B*15, EMBED_DIM+394)

        # 2-layer MLP with LayerNorm (matches PokemonShowdownClaude TokenProjection)
        tok_flat = nn.Dense(self.d_model * 2)(tok_flat)
        tok_flat = nn.LayerNorm()(tok_flat)
        tok_flat = nn.gelu(tok_flat)
        tok_flat = nn.Dense(self.d_model)(tok_flat)
        tok_flat = nn.LayerNorm()(tok_flat)
        tok = tok_flat.reshape(B, N_TOKENS, self.d_model)              # (B, 15, d_model)

        # --- Positional embeddings ---
        pos_emb = nn.Embed(N_TOKENS, self.d_model)(
            jnp.arange(N_TOKENS, dtype=jnp.int32)
        )  # (15, d_model)
        tok = tok + pos_emb[None, :, :]  # broadcast over batch

        # --- Actor / Critic query tokens (learned, independent of input) ---
        actor_query  = self.param("actor_query",
                                  nn.initializers.normal(0.02),
                                  (self.d_model,))
        critic_query = self.param("critic_query",
                                  nn.initializers.normal(0.02),
                                  (self.d_model,))
        # Zero out projected input for query tokens, replace with learned vector
        tok = tok.at[:, ACTOR_IDX, :].set(
            jnp.broadcast_to(actor_query[None, :], (B, self.d_model))
        )
        tok = tok.at[:, CRITIC_IDX, :].set(
            jnp.broadcast_to(critic_query[None, :], (B, self.d_model))
        )

        # --- Transformer ---
        attn_bias = _build_attn_bias()  # (1, 1, 15, 15)
        for _ in range(self.n_layers):
            tok = TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
            )(tok, attn_bias, deterministic=deterministic)

        # --- Heads ---
        actor_out  = tok[:, ACTOR_IDX]    # (B, d_model)
        critic_out = tok[:, CRITIC_IDX]   # (B, d_model)

        # Policy head
        logits = nn.Dense(self.n_actions)(actor_out)          # (B, n_actions)
        safe_legal = jnp.where(legal_mask.sum(-1, keepdims=True) > 0,
                               legal_mask, jnp.ones_like(legal_mask))
        logits = jnp.where(safe_legal > 0, logits, logits - 1e9)
        log_probs = jax.nn.log_softmax(logits, axis=-1)       # (B, n_actions)

        # Distributional value head (C51)
        value_logits = nn.Dense(self.n_atoms)(critic_out)     # (B, n_atoms)
        value_probs  = jax.nn.softmax(value_logits, axis=-1)  # (B, n_atoms)
        support = jnp.linspace(self.v_min, self.v_max, self.n_atoms)
        value   = (value_probs * support[None, :]).sum(-1)    # (B,)

        return log_probs, value_probs, value


# ---------------------------------------------------------------------------
# Convenience: model config as a plain dict (for init calls)
# ---------------------------------------------------------------------------

MODEL_CONFIG = dict(
    d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
    n_actions=N_ACTIONS, n_atoms=N_ATOMS, v_min=V_MIN, v_max=V_MAX,
    n_species=N_SPECIES, n_moves=N_MOVES, n_abilities=N_ABILITIES, n_items=N_ITEMS,
    species_embed_dim=SPECIES_EMBED_DIM, move_embed_dim=MOVE_EMBED_DIM,
    ability_embed_dim=ABILITY_EMBED_DIM, item_embed_dim=ITEM_EMBED_DIM,
    embed_dim=EMBED_DIM,
)
