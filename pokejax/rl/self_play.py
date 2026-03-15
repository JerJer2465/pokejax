"""
Self-play trainer: wraps PPO + rollout into a training loop.

Both players use the same shared model (symmetric self-play).
The model is updated after each rollout using PPO.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import os
import pickle
import time

import jax
import jax.numpy as jnp
import optax

from pokejax.rl.model import PokeTransformer, MODEL_CONFIG
from pokejax.rl.ppo import PPOConfig, TrainState, create_train_state, ppo_step, make_jit_ppo_epochs
from pokejax.rl.rollout import RolloutConfig, collect_rollout, make_jit_rollout, RolloutBatch


@dataclass
class TrainConfig:
    ppo:     PPOConfig     = field(default_factory=PPOConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    total_timesteps: int   = 100_000_000
    log_interval:    int   = 10      # PPO updates between logs
    checkpoint_interval: int = 100   # PPO updates between checkpoints
    checkpoint_dir:  str   = "checkpoints"


def create_model_and_state(
    cfg: TrainConfig,
    key: jnp.ndarray,
    init_params: Optional[dict] = None,
) -> tuple:
    """Initialize model parameters and optimizer state.

    If init_params is provided (e.g. from BC), use those instead of random init.
    """
    model = PokeTransformer()

    if init_params is None:
        # Random init
        B = 1
        dummy_int_ids    = jnp.zeros((B, 15, 8), dtype=jnp.int32)
        dummy_float_feats = jnp.zeros((B, 15, 394), dtype=jnp.float32)
        dummy_legal      = jnp.ones((B, 10), dtype=jnp.float32)
        params = model.init(key, dummy_int_ids, dummy_float_feats, dummy_legal)
    else:
        params = init_params

    lr_schedule = optax.cosine_decay_schedule(
        cfg.ppo.lr, decay_steps=cfg.total_timesteps
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.ppo.max_grad_norm),
        optax.adam(lr_schedule),
    )

    train_state = TrainState(
        params=params,
        opt_state=optimizer.init(params),
        step=jnp.int32(0),
    )
    return model, optimizer, train_state


def train(
    env,
    tables,
    cfg: TrainConfig,
    key: jnp.ndarray,
    init_params: Optional[dict] = None,
) -> TrainState:
    """Main training loop.

    env:         PokeJAXEnv instance
    tables:      Tables object
    cfg:         TrainConfig
    key:         JAX PRNGKey
    init_params: Optional pre-trained params (from BC) to warm-start
    """
    key, init_key = jax.random.split(key)
    model, optimizer, train_state = create_model_and_state(cfg, init_key, init_params)

    # JIT compile rollout and PPO epochs once — reused every iteration.
    # ppo_epochs fuses all n_epochs × n_minibatches into one XLA kernel.
    jit_rollout = make_jit_rollout(model, env, tables, cfg.rollout)
    jit_epochs  = make_jit_ppo_epochs(model, optimizer, cfg.ppo)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    timesteps_collected = 0
    update_idx = 0
    metrics = {}
    t_start = time.time()

    transitions_per_rollout = cfg.rollout.n_envs * cfg.rollout.n_steps

    print(f"PPO Self-Play Training:")
    print(f"  Envs: {cfg.rollout.n_envs}, Steps/rollout: {cfg.rollout.n_steps}")
    print(f"  Transitions/update: {transitions_per_rollout:,}")
    print(f"  Total timesteps: {cfg.total_timesteps:,}")
    if init_params is not None:
        print(f"  Warm-started from pre-trained params")
    print()

    while timesteps_collected < cfg.total_timesteps:
        key, rollout_key, ppo_key = jax.random.split(key, 3)

        # Collect transitions (single fused XLA kernel)
        _, batch = jit_rollout(train_state.params, rollout_key)

        # PPO epochs: all epochs × minibatches in one XLA dispatch
        train_state, metrics, _ = jit_epochs(train_state, batch, ppo_key)

        timesteps_collected += transitions_per_rollout
        update_idx += 1

        if update_idx % cfg.log_interval == 0:
            elapsed = time.time() - t_start
            sps = timesteps_collected / max(elapsed, 1)
            print(f"[{timesteps_collected:>10,}] "
                  f"loss={float(metrics['loss/total']):.4f}  "
                  f"policy={float(metrics['loss/policy']):.4f}  "
                  f"value={float(metrics['loss/value']):.4f}  "
                  f"ent={float(metrics['loss/entropy']):.4f}  "
                  f"SPS={sps:,.0f}")

        # Checkpoint
        if update_idx % cfg.checkpoint_interval == 0:
            ckpt = {
                "params": train_state.params,
                "step": update_idx,
                "timesteps": timesteps_collected,
            }
            path = os.path.join(cfg.checkpoint_dir, f"ppo_{update_idx:06d}.pkl")
            with open(path, "wb") as f:
                pickle.dump(ckpt, f)
            # Also save as latest
            latest_path = os.path.join(cfg.checkpoint_dir, "ppo_latest.pkl")
            with open(latest_path, "wb") as f:
                pickle.dump(ckpt, f)
            print(f"  Checkpoint saved: {path}")

    # Final checkpoint
    ckpt = {
        "params": train_state.params,
        "step": update_idx,
        "timesteps": timesteps_collected,
    }
    path = os.path.join(cfg.checkpoint_dir, "ppo_final.pkl")
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)
    print(f"Final checkpoint: {path}")

    return train_state
