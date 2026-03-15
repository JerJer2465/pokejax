#!/usr/bin/env python3
"""
Train PokeJAX bot via PPO self-play.

Usage:
    # First run (XLA compiles ~10 min, then cached):
    python scripts/train_ppo.py

    # With team pool:
    python scripts/train_ppo.py --team-pool data/team_pool.npz

    # Initialize from BC checkpoint:
    python scripts/train_ppo.py --team-pool data/team_pool.npz --bc-init checkpoints/bc_best.pkl

    # Resume from PPO checkpoint:
    python scripts/train_ppo.py --checkpoint checkpoints/ppo_latest.pkl
"""

import argparse
import os
import sys
import pickle

sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1, closefd=False)

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
    parser.add_argument("--bc-init",     type=str,   default=None,
                        help="Initialize from BC checkpoint .pkl")
    parser.add_argument("--checkpoint",  type=str,   default=None,
                        help="Resume from PPO checkpoint .pkl")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=100,
                        help="Save checkpoint every N PPO updates")
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
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
    )

    # Load initial params from BC or PPO checkpoint
    init_params = None
    if args.bc_init and os.path.exists(args.bc_init):
        print(f"Loading BC checkpoint: {args.bc_init}")
        with open(args.bc_init, "rb") as f:
            init_params = pickle.load(f)["params"]
        print("  BC params loaded (fresh optimizer state)")
    elif args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading PPO checkpoint: {args.checkpoint}")
        with open(args.checkpoint, "rb") as f:
            init_params = pickle.load(f)["params"]
        print("  PPO params loaded")

    key = jax.random.PRNGKey(args.seed)
    print("Starting PPO self-play training...")
    final_state = train(env, env.tables, cfg, key, init_params=init_params)
    print("Training complete.")


if __name__ == "__main__":
    main()
