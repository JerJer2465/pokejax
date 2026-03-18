"""
Vectorized rollout: vmap(env) × lax.scan(steps) = full XLA fusion.

PERFORMANCE OPTIMIZATIONS (vs original):
  1. Batched model forward: both players' obs are stacked into a single
     batch-2 model call (for symmetric self-play), halving transformer cost.
  2. Lean env step: uses env.step_no_obs() to skip redundant observation
     building inside env.step() (obs are already built by the rollout).
  3. Auto-reset: finished environments are immediately reset so every
     scan step produces useful training data.

The key structure:
    vmap over N_ENVS environments (parallel battles)
    lax.scan over N_STEPS per rollout
    → N_STEPS × N_ENVS transitions per PPO batch, all on-device

Usage:
    from pokejax.rl.rollout import collect_rollout, RolloutConfig

    cfg = RolloutConfig()
    carry, batch, info = collect_rollout(model, params, env, tables, cfg, key)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple, Optional

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
    gamma:      float = 0.999   # match PPOConfig default
    gae_lambda: float = 0.95
    player:     int   = 0       # which player's perspective to collect for PPO


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


def _sample_action_batched(
    model,
    params: dict,
    obs_p0: dict,
    obs_p1: dict,
    key_p0: jnp.ndarray,
    key_p1: jnp.ndarray,
) -> tuple:
    """Sample actions for BOTH players in a single batch-2 model forward pass.

    Returns (act_p0, lp_p0, val_p0, act_p1, lp_p1, val_p1).
    """
    # Stack both players' observations into batch dim
    int_ids_2 = jnp.stack([obs_p0["int_ids"], obs_p1["int_ids"]])        # (2, 15, 8)
    float_feats_2 = jnp.stack([obs_p0["float_feats"], obs_p1["float_feats"]])  # (2, 15, 394)
    legal_mask_2 = jnp.stack([obs_p0["legal_mask"], obs_p1["legal_mask"]])      # (2, 10)

    # Single forward pass for both players
    log_probs_2, _, values_2 = model.apply(params, int_ids_2, float_feats_2, legal_mask_2)
    # log_probs_2: (2, 10), values_2: (2,)

    # Sample actions
    act_p0 = jax.random.categorical(key_p0, log_probs_2[0])
    lp_p0 = log_probs_2[0, act_p0]
    val_p0 = values_2[0]

    act_p1 = jax.random.categorical(key_p1, log_probs_2[1])
    lp_p1 = log_probs_2[1, act_p1]
    val_p1 = values_2[1]

    return act_p0, lp_p0, val_p0, act_p1, lp_p1, val_p1


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


def make_rollout_step_fn(model, params, opp_params, env, tables, cfg: RolloutConfig):
    """Return a lax.scan-compatible step function.

    params:     training agent's params (player cfg.player)
    opp_params: opponent's params (player 1-cfg.player). If same pytree as
                params, this is symmetric self-play.
    """
    # Detect symmetric self-play (same params object) to enable batched forward
    is_symmetric = params is opp_params

    def _step(carry: StepCarry, _) -> tuple[StepCarry, StepOutput]:
        env_state, key = carry
        key, act_key_p0, act_key_p1, step_key, reset_key = jax.random.split(key, 5)

        # Build obs for both players
        obs_p0 = _obs_from_env_state(env_state, player=0, tables=tables)
        obs_p1 = _obs_from_env_state(env_state, player=1, tables=tables)

        if is_symmetric:
            # Batched forward pass: both players in one model call
            act_p0, lp_p0, val_p0, act_p1, lp_p1, val_p1 = _sample_action_batched(
                model, params, obs_p0, obs_p1, act_key_p0, act_key_p1,
            )
        else:
            # Asymmetric: separate forward passes with different params
            p0_params = params if cfg.player == 0 else opp_params
            p1_params = opp_params if cfg.player == 0 else params
            act_p0, lp_p0, val_p0 = _sample_action(model, p0_params, obs_p0, act_key_p0)
            act_p1, lp_p1, val_p1 = _sample_action(model, p1_params, obs_p1, act_key_p1)

        actions = jnp.array([act_p0, act_p1], dtype=jnp.int32)

        # Environment step — use step_lean to skip redundant obs building
        new_env_state, rewards, dones = env.step_lean(
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

        # Auto-reset: if episode finished, reset to new battle
        reset_state, _ = env.reset(reset_key)
        final_env_state = jax.tree.map(
            lambda r, n: jnp.where(done, r, n),
            reset_state, new_env_state,
        )

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

        return StepCarry(env_state=final_env_state, key=key), output

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
    opp_params: Optional[dict] = None,
) -> tuple[StepCarry, RolloutBatch, dict]:
    """
    Collect n_envs × n_steps transitions using vmap + lax.scan.

    Args:
        opp_params: If provided, opponent uses these params instead of `params`.
                    Enables historical pool / frozen opponent self-play.

    Returns (final_carry, RolloutBatch, info) where:
        - batch arrays have shape (n_steps * n_envs, ...)
        - info contains episode tracking: rewards, dones per step per env
    """
    if opp_params is None:
        opp_params = params  # symmetric self-play

    # Split keys for each env
    env_keys = jax.random.split(key, cfg.n_envs + 1)
    key, env_keys = env_keys[0], env_keys[1:]

    # Reset all envs
    v_reset = jax.vmap(env.reset)
    init_env_states, _ = v_reset(env_keys)

    # Build per-env step function
    step_fn = make_rollout_step_fn(model, params, opp_params, env, tables, cfg)

    # lax.scan over steps for each env (scan inside vmap)
    def scan_env(carry: StepCarry, _) -> tuple[StepCarry, StepOutput]:
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
        values_ext = jnp.append(values_traj, boot_val)  # (n_steps+1,)
        return compute_gae(rewards, values_ext, dones, cfg.gamma, cfg.gae_lambda)

    advantages, returns = jax.vmap(_gae_one)(
        traj.reward,
        traj.value,
        traj.done.astype(jnp.float32),
        boot_values,
    )

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

    # Episode stats as scalars — computed inside JIT, no extra GPU memory
    done_f = traj.done.astype(jnp.float32)        # (n_envs, n_steps)
    terminal_rewards = traj.reward * done_f         # reward only at episode end
    n_episodes = done_f.sum()
    n_wins = (terminal_rewards > 0).astype(jnp.float32).sum()
    mean_reward = jnp.where(n_episodes > 0, terminal_rewards.sum() / n_episodes, 0.0)
    win_rate = jnp.where(n_episodes > 0, n_wins / n_episodes, 0.5)

    info = {
        "n_episodes":  n_episodes,
        "win_rate":    win_rate,
        "mean_reward": mean_reward,
    }

    return final_carries, batch, info


# ---------------------------------------------------------------------------
# Convenience: pre-bake static args and return jitted rollout functions
# ---------------------------------------------------------------------------

def make_jit_rollout(model, env, tables, cfg: RolloutConfig):
    """
    Return a jax.jit-compiled rollout function with signature:
        jit_rollout(params, key) -> (StepCarry, RolloutBatch, info)

    Symmetric self-play only (both players use same params).
    """
    @jax.jit
    def _jit_rollout(params: dict, key: jnp.ndarray):
        return collect_rollout(model, params, env, tables, cfg, key)

    return _jit_rollout


def make_jit_rollout_asymmetric(model, env, tables, cfg: RolloutConfig):
    """
    Return a jax.jit-compiled rollout function with signature:
        jit_rollout(params, opp_params, key) -> (StepCarry, RolloutBatch, info)

    Asymmetric self-play: training agent uses `params`, opponent uses `opp_params`.
    """
    @jax.jit
    def _jit_rollout(params: dict, opp_params: dict, key: jnp.ndarray):
        return collect_rollout(model, params, env, tables, cfg, key, opp_params=opp_params)

    return _jit_rollout


