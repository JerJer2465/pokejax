"""
Self-play trainer: wraps PPO + rollout into a training loop.

Both players use the same shared model (symmetric self-play).
The model is updated after each rollout using PPO.
"""

from __future__ import annotations
from dataclasses import dataclass, field

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
    checkpoint_dir:  str   = "/tmp/pokejax_checkpoints"


def create_model_and_state(cfg: TrainConfig, key: jnp.ndarray) -> tuple:
    """Initialize model parameters and optimizer state."""
    model = PokeTransformer()

    # Dummy inputs for init
    B = 1
    dummy_int_ids    = jnp.zeros((B, 15, 8), dtype=jnp.int32)
    dummy_float_feats = jnp.zeros((B, 15, 394), dtype=jnp.float32)
    dummy_legal      = jnp.ones((B, 10), dtype=jnp.float32)

    params = model.init(key, dummy_int_ids, dummy_float_feats, dummy_legal)

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
) -> TrainState:
    """Main training loop.

    env:    PokeJAXEnv instance
    tables: Tables object
    cfg:    TrainConfig
    key:    JAX PRNGKey
    """
    key, init_key = jax.random.split(key)
    model, optimizer, train_state = create_model_and_state(cfg, init_key)

    # JIT compile rollout and PPO epochs once — reused every iteration.
    # ppo_epochs fuses all n_epochs × n_minibatches into one XLA kernel.
    jit_rollout = make_jit_rollout(model, env, tables, cfg.rollout)
    jit_epochs  = make_jit_ppo_epochs(model, optimizer, cfg.ppo)

    timesteps_collected = 0
    update_idx = 0
    metrics = {}

    transitions_per_rollout = cfg.rollout.n_envs * cfg.rollout.n_steps

    while timesteps_collected < cfg.total_timesteps:
        key, rollout_key, ppo_key = jax.random.split(key, 3)

        # Collect transitions (single fused XLA kernel)
        _, batch = jit_rollout(train_state.params, rollout_key)

        # PPO epochs: all epochs × minibatches in one XLA dispatch
        train_state, metrics, _ = jit_epochs(train_state, batch, ppo_key)

        timesteps_collected += transitions_per_rollout
        update_idx += 1

        if update_idx % cfg.log_interval == 0:
            print(f"[{timesteps_collected:>10,}] "
                  f"loss={float(metrics['loss/total']):.4f}  "
                  f"policy={float(metrics['loss/policy']):.4f}  "
                  f"value={float(metrics['loss/value']):.4f}  "
                  f"ent={float(metrics['loss/entropy']):.4f}")

    return train_state
