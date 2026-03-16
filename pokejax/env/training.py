"""
PPO-compatible training utilities for PokeJAX.

Provides vectorized rollout collection using vmap + lax.scan for maximum
GPU throughput. Designed to drop into existing PPO training loops.

Usage:
    from pokejax.env.training import make_rollout_fn

    env = PokeJAXEnv(gen=4, team_pool=pool)
    rollout_fn = make_rollout_fn(env, n_envs=1024, n_steps=128)

    # In training loop:
    key = jax.random.PRNGKey(0)
    states, key = init_states(env, n_envs, key)
    rollout_data, states, key = rollout_fn(states, policy_fn, key)
"""

import jax
import jax.numpy as jnp
import functools

from pokejax.types import BattleState


def init_states(env, n_envs: int, key: jnp.ndarray):
    """Initialize n_envs parallel battle states."""
    keys = jax.random.split(key, n_envs + 1)
    key = keys[0]
    env_keys = keys[1:]
    v_reset = jax.vmap(env.reset)
    states, obss = v_reset(env_keys)
    return states, obss, key


def make_rollout_fn(env, n_envs: int, n_steps: int):
    """
    Create a JIT-compiled rollout function that collects n_steps of experience
    across n_envs parallel environments.

    Args:
        env: PokeJAXEnv instance
        n_envs: number of parallel environments
        n_steps: rollout length per collection

    Returns:
        rollout_fn(states, policy_fn, key) -> (rollout_data, new_states, new_key)

        where rollout_data is a dict with:
            obs:          float32[n_steps, n_envs, 2, OBS_DIM]
            actions:      int32[n_steps, n_envs, 2]
            rewards:      float32[n_steps, n_envs, 2]
            dones:        bool[n_steps, n_envs, 2]
            action_masks: bool[n_steps, n_envs, 2, N_ACTIONS]
            values:       float32[n_steps, n_envs, 2]  (if policy returns values)
            log_probs:    float32[n_steps, n_envs, 2]
    """

    v_step = jax.vmap(env.step_autoreset)
    v_masks = jax.vmap(env.get_action_masks)
    v_reset = jax.vmap(env.reset)

    def _rollout(states, policy_fn, key):
        """
        Collect a rollout of n_steps across n_envs environments.

        policy_fn(obs, action_mask, key) -> (actions, log_probs, values, new_key)
            obs:         float32[n_envs, 2, OBS_DIM]
            action_mask: bool[n_envs, 2, N_ACTIONS]
            returns:
                actions:   int32[n_envs, 2]
                log_probs: float32[n_envs, 2]
                values:    float32[n_envs, 2]
        """
        # Get initial observations
        obs0_p0 = jax.vmap(
            lambda s: jax.vmap(
                lambda p: jnp.zeros(env.obs_dim, dtype=jnp.float32)
            )(jnp.arange(2))
        )(states)  # placeholder, will be overwritten

        # Build initial obs from states
        v_obs = jax.vmap(
            lambda s: jnp.stack([
                env._build_obs_static(s, 0),
                env._build_obs_static(s, 1),
            ], axis=0)
        ) if hasattr(env, '_build_obs_static') else None

        def _scan_step(carry, _):
            states, key = carry

            # Get action masks for all envs
            masks = v_masks(states)  # [n_envs, 2, N_ACTIONS]

            # Build observations
            obs_p0 = jax.vmap(
                lambda s: build_observation(s, player=0, tables=env.tables)
            )(states)
            obs_p1 = jax.vmap(
                lambda s: build_observation(s, player=1, tables=env.tables)
            )(states)
            obs = jnp.stack([obs_p0, obs_p1], axis=1)  # [n_envs, 2, OBS_DIM]

            # Get actions from policy
            key, policy_key = jax.random.split(key)
            actions, log_probs, values = policy_fn(obs, masks, policy_key)

            # Step all envs
            key, *step_keys = jax.random.split(key, n_envs + 1)
            step_keys = jnp.stack(step_keys)
            new_states, new_obs, rewards, dones, _ = v_step(
                states, actions, step_keys
            )

            transition = {
                'obs': obs,
                'actions': actions,
                'rewards': rewards,
                'dones': dones,
                'action_masks': masks,
                'log_probs': log_probs,
                'values': values,
            }

            return (new_states, key), transition

        (final_states, final_key), rollout = jax.lax.scan(
            _scan_step, (states, key), None, length=n_steps
        )

        return rollout, final_states, final_key

    return jax.jit(_rollout, static_argnums=(1,))


def make_random_rollout_fn(env, n_envs: int, n_steps: int):
    """
    Create a rollout function with random legal actions (for smoke testing).
    No policy network needed.

    Returns:
        rollout_fn(states, key) -> (rollout_data, new_states, new_key)
    """
    from pokejax.env.obs import build_observation

    v_step = jax.vmap(env.step_autoreset)
    v_masks = jax.vmap(env.get_action_masks)

    def _random_rollout(states, key):
        def _scan_step(carry, _):
            states, key = carry

            # Get action masks
            masks = v_masks(states)  # [n_envs, 2, N_ACTIONS]

            # Sample random legal actions for each env and each player
            key, k0, k1 = jax.random.split(key, 3)

            def _sample_legal(mask, subkey):
                # Mask illegal actions, sample uniformly from legal ones
                logits = jnp.where(mask, jnp.float32(0.0), jnp.float32(-1e9))
                return jax.random.categorical(subkey, logits).astype(jnp.int32)

            keys_p0 = jax.random.split(k0, n_envs)
            keys_p1 = jax.random.split(k1, n_envs)
            a0 = jax.vmap(_sample_legal)(masks[:, 0], keys_p0)  # [n_envs]
            a1 = jax.vmap(_sample_legal)(masks[:, 1], keys_p1)  # [n_envs]
            actions = jnp.stack([a0, a1], axis=1)  # [n_envs, 2]

            # Step all envs
            key, *step_keys = jax.random.split(key, n_envs + 1)
            step_keys = jnp.stack(step_keys)
            new_states, obs, rewards, dones, _ = v_step(states, actions, step_keys)

            transition = {
                'actions': actions,
                'rewards': rewards,
                'dones': dones,
            }

            return (new_states, key), transition

        (final_states, final_key), rollout = jax.lax.scan(
            _scan_step, (states, key), None, length=n_steps
        )

        return rollout, final_states, final_key

    return jax.jit(_random_rollout)
