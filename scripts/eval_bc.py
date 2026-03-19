#!/usr/bin/env python3
"""
Evaluate a BC/PPO checkpoint vs random and/or heuristic opponents.

Usage:
    python scripts/eval_bc.py --checkpoint checkpoints/bc_final.pkl --games 50
"""

import argparse
import os
import sys
import pickle
import time

sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1, closefd=False)

import numpy as np
import jax
import jax.numpy as jnp


def main():
    parser = argparse.ArgumentParser(description="Evaluate PokeJAX checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--gen", type=int, default=4)
    parser.add_argument("--team-pool", type=str, default=None)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed", type=int, default=9999)
    parser.add_argument("--vs", type=str, default="both",
                        choices=["random", "heuristic", "both"])
    args = parser.parse_args()

    print(f"JAX backend: {jax.default_backend()}")

    from pokejax.env.pokejax_env import PokeJAXEnv
    from pokejax.rl.model import create_model
    from pokejax.rl.bc import eval_vs_random, eval_vs_heuristic

    env = PokeJAXEnv(gen=args.gen, team_pool_path=args.team_pool)

    print(f"Loading checkpoint: {args.checkpoint}")
    with open(args.checkpoint, "rb") as f:
        ckpt = pickle.load(f)
    arch = ckpt.get("arch", "transformer")
    model = create_model(arch)
    params = ckpt["params"]
    step = ckpt.get("step", "?")
    print(f"  Step: {step}")

    if args.vs in ("random", "both"):
        print(f"\nEvaluating vs Random ({args.games} games)...")
        t0 = time.time()
        result = eval_vs_random(model, params, env, n_games=args.games, seed=args.seed)
        elapsed = time.time() - t0
        print(f"  Win rate: {result['win_rate']:.1%}")
        print(f"  Avg turns: {result['avg_turns']:.1f}")
        print(f"  Time: {elapsed:.1f}s ({args.games / elapsed:.1f} games/s)")

    if args.vs in ("heuristic", "both"):
        print(f"\nEvaluating vs Heuristic ({args.games} games)...")
        t0 = time.time()
        result = eval_vs_heuristic(model, params, env, n_games=args.games, seed=args.seed + 1)
        elapsed = time.time() - t0
        print(f"  Win rate: {result['win_rate']:.1%}")
        print(f"  Avg turns: {result['avg_turns']:.1f}")
        print(f"  Time: {elapsed:.1f}s ({args.games / elapsed:.1f} games/s)")


if __name__ == "__main__":
    main()
