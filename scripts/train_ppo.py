#!/usr/bin/env python3
"""
Train PokeJAX bot via PPO self-play.

Usage:
    # First run (XLA compiles ~10 min, then cached):
    python scripts/train_ppo.py

    # With team pool:
    python scripts/train_ppo.py --team-pool data/team_pool.npz

    # Resume from checkpoint:
    python scripts/train_ppo.py --checkpoint /tmp/pokejax_checkpoints/latest
"""

import argparse
import os

import jax
import jax.numpy as jnp

# Enable persistent XLA compilation cache — first run ~10 min, subsequent runs instant
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_compile_cache")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen",         type=int,   default=4)
    parser.add_argument("--team-pool",   type=str,   default=None)
    parser.add_argument("--n-envs",      type=int,   default=512)
    parser.add_argument("--n-steps",     type=int,   default=128)
    parser.add_argument("--total-steps", type=int,   default=100_000_000)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--checkpoint",  type=str,   default=None)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")

    from pokejax.env.pokejax_env import PokeJAXEnv
    from pokejax.rl.self_play import TrainConfig, RolloutConfig, PPOConfig, train

    env = PokeJAXEnv(gen=args.gen, team_pool_path=args.team_pool)

    cfg = TrainConfig(
        ppo=PPOConfig(lr=args.lr),
        rollout=RolloutConfig(n_envs=args.n_envs, n_steps=args.n_steps),
        total_timesteps=args.total_steps,
    )

    key = jax.random.PRNGKey(args.seed)
    print("Starting training...")
    final_state = train(env, env.tables, cfg, key)
    print("Training complete.")


if __name__ == "__main__":
    main()
