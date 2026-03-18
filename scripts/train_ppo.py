#!/usr/bin/env python3
"""
Train PokeJAX bot via PPO self-play with TensorBoard logging.

Usage:
    # First run (XLA compiles ~10 min, then cached):
    python scripts/train_ppo.py

    # With BC init (recommended):
    python scripts/train_ppo.py --bc-init checkpoints/bc_best.pkl

    # Full configuration:
    python scripts/train_ppo.py \
        --bc-init checkpoints/bc_best.pkl \
        --n-envs 512 --n-steps 128 \
        --total-steps 250000000 \
        --lr 1e-4 --gamma 0.999 --ent-coef 0.02 \
        --pool-size 20 --pool-latest-ratio 0.75 \
        --eval-interval 50 --eval-games 64 \
        --tensorboard-dir runs

    # Resume from PPO checkpoint:
    python scripts/train_ppo.py --checkpoint checkpoints/ppo_latest.pkl

    # View TensorBoard:
    tensorboard --logdir runs --host 0.0.0.0
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
    parser = argparse.ArgumentParser(
        description="PPO self-play training for PokeJAX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Environment
    parser.add_argument("--gen",         type=int,   default=4)
    parser.add_argument("--team-pool",   type=str,   default=None)

    # Training scale
    parser.add_argument("--n-envs",      type=int,   default=512)
    parser.add_argument("--n-steps",     type=int,   default=128)
    parser.add_argument("--total-steps", type=int,   default=250_000_000)

    # PPO hyperparameters
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--gamma",       type=float, default=0.999)
    parser.add_argument("--gae-lambda",  type=float, default=0.95)
    parser.add_argument("--clip-eps",    type=float, default=0.2)
    parser.add_argument("--ent-coef",    type=float, default=0.02)
    parser.add_argument("--vf-coef",     type=float, default=0.5)
    parser.add_argument("--n-epochs",    type=int,   default=2)
    parser.add_argument("--minibatch-size", type=int, default=8192)
    parser.add_argument("--lr-warmup",   type=int,   default=1000,
                        help="Linear LR warmup steps")

    # Opponent pool
    parser.add_argument("--pool-size",   type=int,   default=20)
    parser.add_argument("--pool-save-interval", type=int, default=50)
    parser.add_argument("--pool-latest-ratio",  type=float, default=0.75,
                        help="Probability of self-play vs pool opponent")

    # Eval
    parser.add_argument("--eval-interval", type=int, default=50,
                        help="Eval every N PPO updates")
    parser.add_argument("--eval-games",  type=int,   default=64,
                        help="Number of eval games per opponent type")

    # Checkpoints
    parser.add_argument("--bc-init",     type=str,   default=None,
                        help="Initialize from BC checkpoint .pkl")
    parser.add_argument("--checkpoint",  type=str,   default=None,
                        help="Resume from PPO checkpoint .pkl")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=100)

    # Logging
    parser.add_argument("--tensorboard-dir", type=str, default="runs")
    parser.add_argument("--print-interval",  type=int, default=10)

    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")

    from pokejax.env.pokejax_env import PokeJAXEnv
    from pokejax.rl.self_play import TrainConfig, PPOConfig, RolloutConfig, train

    env = PokeJAXEnv(gen=args.gen, team_pool_path=args.team_pool)

    cfg = TrainConfig(
        ppo=PPOConfig(
            lr=args.lr,
            clip_eps=args.clip_eps,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            n_epochs=args.n_epochs,
            minibatch_size=args.minibatch_size,
        ),
        rollout=RolloutConfig(
            n_envs=args.n_envs,
            n_steps=args.n_steps,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        ),
        total_timesteps=args.total_steps,
        log_interval=1,
        print_interval=args.print_interval,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        tensorboard_dir=args.tensorboard_dir,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        pool_size=args.pool_size,
        pool_save_interval=args.pool_save_interval,
        pool_latest_ratio=args.pool_latest_ratio,
        lr_warmup_steps=args.lr_warmup,
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
