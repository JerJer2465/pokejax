"""
JAX PPO with C51 distributional value + GAE.

Implements clipped PPO updates over batches of rollout transitions.
All operations are pure JAX so this can run inside jax.jit.

Usage:
    ppo = PPOConfig()
    train_state = create_train_state(model, params, ppo)
    train_state, metrics = ppo_update(train_state, batch, ppo)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    # PPO — tuned for Pokemon battles (medium-length episodes, complex action space)
    # References: ps-ppo (Nebraskinator), Karten et al. 2025, 37 PPO details
    lr:              float = 3e-4    # tuned for ~2.2M param model
    clip_eps:        float = 0.2
    gamma:           float = 0.999   # high discount: early moves matter for win/loss
    gae_lambda:      float = 0.95
    vf_coef:         float = 1.0     # increased: value head needs stronger gradient signal for C51
    ent_coef:        float = 0.02    # starting entropy for exploration in complex action space
    ent_coef_end:    float = 0.005   # raised floor: prevents premature convergence
    ent_coef_decay_steps: int = 15000  # optimizer steps over which to anneal entropy
    max_grad_norm:   float = 0.5
    n_epochs:        int   = 3       # 3 epochs: extract more learning per rollout (KL/clip still safe)
    minibatch_size:  int   = 8192    # larger minibatch: better GPU utilization, fewer steps

    # C51
    n_atoms:   int   = 51
    v_min:     float = -1.5
    v_max:     float =  1.5


# ---------------------------------------------------------------------------
# Rollout batch (collected by rollout.py)
# ---------------------------------------------------------------------------

class RolloutBatch(NamedTuple):
    """A batch of transitions for one PPO update epoch.

    All arrays have leading dim = (n_steps * n_envs).
    """
    int_ids:    jnp.ndarray   # (T, 15, 8)      int
    float_feats: jnp.ndarray  # (T, 15, 394)    float32
    legal_mask: jnp.ndarray   # (T, 10)         float32
    actions:    jnp.ndarray   # (T,)            int32
    log_probs_old: jnp.ndarray # (T,)           float32
    advantages: jnp.ndarray   # (T,)            float32
    returns:    jnp.ndarray   # (T,)            float32  (GAE returns)
    dones:      jnp.ndarray   # (T,)            bool


# ---------------------------------------------------------------------------
# Train state (params + optimizer state)
# ---------------------------------------------------------------------------

class TrainState(NamedTuple):
    params:    dict
    opt_state: optax.OptState
    step:      jnp.ndarray   # global step counter


def create_train_state(
    model,        # PokeTransformer instance
    params: dict,
    cfg: PPOConfig,
) -> TrainState:
    """Initialize optimizer and return TrainState."""
    lr_schedule = optax.cosine_decay_schedule(cfg.lr, decay_steps=1_000_000)
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(lr_schedule),
    )
    opt_state = optimizer.init(params)
    return TrainState(params=params, opt_state=opt_state, step=jnp.int32(0))


# ---------------------------------------------------------------------------
# GAE (Generalized Advantage Estimation)
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: jnp.ndarray,    # (T,)
    values: jnp.ndarray,     # (T+1,)  last entry = bootstrap value
    dones: jnp.ndarray,      # (T,)    1.0 if terminal
    gamma: float,
    gae_lambda: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute GAE advantages and returns via lax.scan (reverse scan).

    Returns: (advantages (T,), returns (T,))
    """
    T = rewards.shape[0]

    def _step(carry, t):
        gae_next = carry
        delta = rewards[t] + gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * gae_next
        return gae, gae

    _, advantages_rev = jax.lax.scan(
        _step,
        jnp.float32(0.0),
        jnp.arange(T - 1, -1, -1, dtype=jnp.int32),
    )
    advantages = advantages_rev[::-1]
    returns = advantages + values[:T]
    return advantages, returns


# ---------------------------------------------------------------------------
# C51 distributional loss helper
# ---------------------------------------------------------------------------

def c51_loss(
    value_probs: jnp.ndarray,  # (B, n_atoms)  predicted
    returns:     jnp.ndarray,  # (B,)           scalar targets
    n_atoms: int,
    v_min: float,
    v_max: float,
) -> jnp.ndarray:
    """C51 cross-entropy loss (projected Bellman for scalar targets)."""
    B = returns.shape[0]
    targets = jnp.clip(returns, v_min, v_max)
    delta_z = (v_max - v_min) / (n_atoms - 1)
    b = (targets - v_min) / delta_z                        # (B,)
    lower = jnp.floor(b).astype(jnp.int32)
    upper = jnp.ceil(b).astype(jnp.int32)
    lower = jnp.clip(lower, 0, n_atoms - 1)
    upper = jnp.clip(upper, 0, n_atoms - 1)

    frac_upper = b - lower.astype(jnp.float32)
    frac_lower = 1.0 - frac_upper

    # Scatter into target distribution — indices must be 1D (B,)
    batch_idx = jnp.arange(B)
    target_dist = jnp.zeros((B, n_atoms), dtype=jnp.float32)
    target_dist = target_dist.at[batch_idx, lower].add(frac_lower)
    target_dist = target_dist.at[batch_idx, upper].add(frac_upper)
    # target_dist row sums = 1.0

    log_probs = jnp.log(jnp.clip(value_probs, 1e-8, 1.0))
    loss = -(target_dist * log_probs).sum(-1)
    return loss.mean()


# ---------------------------------------------------------------------------
# PPO loss function
# ---------------------------------------------------------------------------

def ppo_loss(
    params: dict,
    model,
    batch: RolloutBatch,
    cfg: PPOConfig,
    step: jnp.ndarray = None,
) -> tuple[jnp.ndarray, dict]:
    """Compute PPO loss and return (total_loss, metrics_dict)."""
    log_probs_new, value_probs, values = model.apply(
        params,
        batch.int_ids,
        batch.float_feats,
        batch.legal_mask,
    )

    # --- Policy loss (PPO-clip) ---
    log_prob_actions = jnp.take_along_axis(
        log_probs_new, batch.actions[:, None], axis=1
    ).squeeze(1)  # (B,)

    ratio = jnp.exp(log_prob_actions - batch.log_probs_old)
    adv = (batch.advantages - batch.advantages.mean()) / (
        batch.advantages.std() + 1e-8
    )
    pg_loss1 = -adv * ratio
    pg_loss2 = -adv * jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
    policy_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

    # --- Value loss (C51) ---
    value_loss = c51_loss(
        value_probs, batch.returns,
        cfg.n_atoms, cfg.v_min, cfg.v_max,
    )

    # --- Entropy bonus (with linear annealing) ---
    entropy = -(jnp.exp(log_probs_new) * log_probs_new).sum(-1).mean()

    if step is not None and cfg.ent_coef_decay_steps > 0:
        frac = jnp.clip(step / cfg.ent_coef_decay_steps, 0.0, 1.0)
        ent_coef = cfg.ent_coef * (1.0 - frac) + cfg.ent_coef_end * frac
    else:
        ent_coef = cfg.ent_coef

    # --- Total ---
    total_loss = (
        policy_loss
        + cfg.vf_coef * value_loss
        - ent_coef * entropy
    )

    # --- Diagnostics ---
    approx_kl = ((ratio - 1.0) - jnp.log(ratio)).mean()
    clip_fraction = (jnp.abs(ratio - 1.0) > cfg.clip_eps).astype(jnp.float32).mean()

    metrics = {
        "loss/total":       total_loss,
        "loss/policy":      policy_loss,
        "loss/value":       value_loss,
        "loss/entropy":     entropy,
        "ratio/mean":       ratio.mean(),
        "ratio/max":        ratio.max(),
        "diagnostics/approx_kl":     approx_kl,
        "diagnostics/clip_fraction": clip_fraction,
        "diagnostics/ent_coef":      ent_coef,
    }
    return total_loss, metrics


# ---------------------------------------------------------------------------
# Single gradient step
# ---------------------------------------------------------------------------

def ppo_step(
    train_state: TrainState,
    model,
    batch: RolloutBatch,
    cfg: PPOConfig,
    optimizer,
) -> tuple[TrainState, dict]:
    """Apply one gradient step. JIT-able."""
    loss_fn = lambda p: ppo_loss(p, model, batch, cfg, step=train_state.step)
    (total_loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        train_state.params
    )
    updates, new_opt_state = optimizer.update(grads, train_state.opt_state,
                                               train_state.params)
    new_params = optax.apply_updates(train_state.params, updates)
    new_state = TrainState(
        params=new_params,
        opt_state=new_opt_state,
        step=train_state.step + jnp.int32(1),
    )
    return new_state, metrics


# ---------------------------------------------------------------------------
# Fused multi-epoch PPO update (lax.scan — single XLA dispatch)
# ---------------------------------------------------------------------------

def ppo_epochs(
    train_state: TrainState,
    model,
    batch: RolloutBatch,
    cfg: PPOConfig,
    optimizer,
    key: jnp.ndarray,
) -> tuple[TrainState, dict, jnp.ndarray]:
    """
    Run `cfg.n_epochs` PPO epochs over `batch`, shuffling each epoch.
    Uses lax.scan so all epochs × minibatches compile to ONE XLA kernel —
    eliminates Python-dispatch overhead vs calling ppo_step in a Python loop.

    Returns (new_train_state, metrics_from_last_mb, new_key).
    """
    B = batch.advantages.shape[0]
    mb_size = cfg.minibatch_size
    n_mb = B // mb_size

    def minibatch_step(carry, mb_idx):
        ts, batch_s = carry
        start = mb_idx * mb_size
        mb = jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(x, start, mb_size, axis=0),
            batch_s,
        )
        ts, metrics = ppo_step(ts, model, mb, cfg, optimizer)
        return (ts, batch_s), metrics

    def epoch_step(carry, _):
        ts, key = carry
        key, perm_key = jax.random.split(key)
        perm = jax.random.permutation(perm_key, B)
        batch_s = jax.tree.map(lambda x: x[perm], batch)
        (ts, _), metrics = jax.lax.scan(
            minibatch_step,
            (ts, batch_s),
            jnp.arange(n_mb, dtype=jnp.int32),
        )
        # Return last-minibatch metrics as epoch summary
        last_metrics = jax.tree.map(lambda m: m[-1], metrics)
        return (ts, key), last_metrics

    (ts, key), all_epoch_metrics = jax.lax.scan(
        epoch_step,
        (train_state, key),
        None,
        length=cfg.n_epochs,
    )
    # Return metrics from final epoch
    final_metrics = jax.tree.map(lambda m: m[-1], all_epoch_metrics)
    return ts, final_metrics, key


def make_jit_ppo_epochs(model, optimizer, cfg: PPOConfig):
    """
    Return a jax.jit-compiled function:
        jit_epochs(train_state, batch, key) -> (TrainState, metrics, key)

    model, optimizer, cfg captured as static closure variables.
    All n_epochs × n_minibatches fused into one XLA kernel.
    """
    @jax.jit
    def _jit_epochs(train_state: TrainState, batch: RolloutBatch, key: jnp.ndarray):
        return ppo_epochs(train_state, model, batch, cfg, optimizer, key)

    return _jit_epochs
