"""
Vectorized rollout: vmap(env) × lax.scan(steps) = full XLA fusion.

The key structure:
    vmap over N_ENVS environments (parallel battles)
    lax.scan over N_STEPS per rollout
    → N_STEPS × N_ENVS transitions per PPO batch, all on-device

Both players share the same model (self-play) and act independently
from their respective perspectives (obs masked by RevealState).

Usage:
    from pokejax.rl.rollout import collect_rollout, RolloutConfig

    cfg = RolloutConfig()
    carry, batch = collect_rollout(model, params, env, tables, cfg, key)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from pokejax.env.pokejax_env import EnvState
from pokejax.rl.obs_builder import build_obs
from pokejax.rl.ppo import RolloutBatch, compute_gae


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RolloutConfig:
    n_envs:     int   = 512
    n_steps:    int   = 128
    gamma:      float = 0.99
    gae_lambda: float = 0.95
    player:     int   = 0    # which player's perspective to collect for PPO


# ---------------------------------------------------------------------------
# Single step: obs → action → env step → next obs
# ---------------------------------------------------------------------------

def _obs_from_env_state(env_state: EnvState, player: int, tables) -> dict:
    """Build observation for one player from EnvState."""
    return build_obs(env_state.battle, env_state.reveal, player, tables)


def _sample_action(
    model,
    params: dict,
    obs: dict,
    key: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample action, return (action, log_prob, value)."""
    log_probs, value_probs, values = model.apply(
        params,
        obs["int_ids"][None],    # add batch dim
        obs["float_feats"][None],
        obs["legal_mask"][None],
    )
    log_probs = log_probs[0]   # remove batch dim
    values    = values[0]

    # Sample from categorical
    action = jax.random.categorical(key, log_probs)
    log_prob = log_probs[action]
    return action, log_prob, values


# ---------------------------------------------------------------------------
# Rollout step (single environment, single step)
# ---------------------------------------------------------------------------

class StepCarry(NamedTuple):
    env_state: EnvState
    key:       jnp.ndarray


class StepOutput(NamedTuple):
    int_ids:    jnp.ndarray   # (15, 8)
    float_feats: jnp.ndarray  # (15, 394)
    legal_mask: jnp.ndarray   # (10,)
    action:     jnp.ndarray   # scalar int32
    log_prob:   jnp.ndarray   # scalar float32
    value:      jnp.ndarray   # scalar float32
    reward:     jnp.ndarray   # scalar float32
    done:       jnp.ndarray   # scalar bool


def make_rollout_step_fn(model, params, env, tables, cfg: RolloutConfig):
    """Return a lax.scan-compatible step function (no Python branches on arrays)."""

    def _step(carry: StepCarry, _) -> tuple[StepCarry, StepOutput]:
        env_state, key = carry
        key, act_key_p0, act_key_p1, step_key = jax.random.split(key, 4)

        # Build obs for both players
        obs_p0 = _obs_from_env_state(env_state, player=0, tables=tables)
        obs_p1 = _obs_from_env_state(env_state, player=1, tables=tables)

        # Sample actions for both players (self-play: shared model)
        act_p0, lp_p0, val_p0 = _sample_action(model, params, obs_p0, act_key_p0)
        act_p1, lp_p1, val_p1 = _sample_action(model, params, obs_p1, act_key_p1)

        actions = jnp.array([act_p0, act_p1], dtype=jnp.int32)

        # Environment step
        new_env_state, _obs_both, rewards, dones, _ = env.step(
            env_state, actions, step_key
        )

        # The PPO agent is always player `cfg.player`
        p = cfg.player
        obs_p = obs_p0 if p == 0 else obs_p1
        reward = rewards[p]
        done   = dones[p]
        lp     = lp_p0 if p == 0 else lp_p1
        val    = val_p0 if p == 0 else val_p1
        act    = act_p0 if p == 0 else act_p1

        output = StepOutput(
            int_ids    = obs_p["int_ids"],
            float_feats= obs_p["float_feats"],
            legal_mask = obs_p["legal_mask"],
            action     = act.astype(jnp.int32),
            log_prob   = lp,
            value      = val,
            reward     = reward,
            done       = done,
        )

        return StepCarry(env_state=new_env_state, key=key), output

    return _step


# ---------------------------------------------------------------------------
# Collect rollout: scan over steps, vmap over envs
# ---------------------------------------------------------------------------

def collect_rollout(
    model,
    params: dict,
    env,           # PokeJAXEnv instance
    tables,
    cfg: RolloutConfig,
    key: jnp.ndarray,
) -> tuple[StepCarry, RolloutBatch]:
    """
    Collect n_envs × n_steps transitions using vmap + lax.scan.

    Returns (final_carry, RolloutBatch) where batch arrays have shape
    (n_steps * n_envs, ...) ready for PPO minibatch sampling.
    """
    # Split keys for each env
    env_keys = jax.random.split(key, cfg.n_envs + 1)
    key, env_keys = env_keys[0], env_keys[1:]

    # Reset all envs
    v_reset = jax.vmap(env.reset)
    init_env_states, _ = v_reset(env_keys)

    # Build per-env step function via vmap
    step_fn = make_rollout_step_fn(model, params, env, tables, cfg)
    v_step_fn = jax.vmap(step_fn, in_axes=(0, None))  # vmap over carry, not xs

    # lax.scan over steps for each env (scan inside vmap)
    def scan_env(carry: StepCarry, _) -> tuple[StepCarry, StepOutput]:
        """Single-env scan step (used inside vmap)."""
        return step_fn(carry, None)

    def collect_one_env(init_carry: StepCarry) -> tuple[StepCarry, StepOutput]:
        final_carry, outputs = jax.lax.scan(
            scan_env, init_carry, None, length=cfg.n_steps
        )
        return final_carry, outputs

    # vmap over environments, then scan over steps
    init_keys_per_env = jax.random.split(key, cfg.n_envs)
    init_carries = jax.vmap(lambda s, k: StepCarry(env_state=s, key=k))(
        init_env_states, init_keys_per_env
    )

    final_carries, traj = jax.vmap(collect_one_env)(init_carries)
    # traj fields shape: (n_envs, n_steps, ...)

    # Bootstrap values for last state of each env
    def _boot_value(carry: StepCarry) -> jnp.ndarray:
        obs = _obs_from_env_state(carry.env_state, cfg.player, tables)
        _, _, val = model.apply(
            params,
            obs["int_ids"][None],
            obs["float_feats"][None],
            obs["legal_mask"][None],
        )
        return val[0]

    boot_values = jax.vmap(_boot_value)(final_carries)  # (n_envs,)

    # Compute GAE for each env
    def _gae_one(rewards, values_traj, dones, boot_val):
        # values_traj: (n_steps,), boot_val: scalar
        values_ext = jnp.append(values_traj, boot_val)  # (n_steps+1,)
        return compute_gae(rewards, values_ext, dones, cfg.gamma, cfg.gae_lambda)

    advantages, returns = jax.vmap(_gae_one)(
        traj.reward,           # (n_envs, n_steps)
        traj.value,            # (n_envs, n_steps)
        traj.done.astype(jnp.float32),
        boot_values,           # (n_envs,)
    )
    # advantages, returns shape: (n_envs, n_steps)

    # Flatten: (n_envs, n_steps, ...) → (n_envs * n_steps, ...)
    def _flat(x):
        return x.reshape(-1, *x.shape[2:])

    batch = RolloutBatch(
        int_ids       = _flat(traj.int_ids),
        float_feats   = _flat(traj.float_feats),
        legal_mask    = _flat(traj.legal_mask),
        actions       = _flat(traj.action),
        log_probs_old = _flat(traj.log_prob),
        advantages    = advantages.reshape(-1),
        returns       = returns.reshape(-1),
        dones         = _flat(traj.done),
    )

    return final_carries, batch


# ---------------------------------------------------------------------------
# Convenience: pre-bake static args and return a jitted rollout function
# ---------------------------------------------------------------------------

def make_jit_rollout(model, env, tables, cfg: RolloutConfig):
    """
    Return a jax.jit-compiled rollout function with signature:
        jit_rollout(params, key) -> (StepCarry, RolloutBatch)

    model, env, tables, cfg are captured as static closure variables so JAX
    traces through them at compile time.  Only params (a pytree) and key
    vary at runtime — no retrace on param updates.

    Usage:
        jit_rollout = make_jit_rollout(model, env, env.tables, cfg)
        carry, batch = jit_rollout(params, key)
    """
    @jax.jit
    def _jit_rollout(params: dict, key: jnp.ndarray):
        return collect_rollout(model, params, env, tables, cfg, key)

    return _jit_rollout
