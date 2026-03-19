#!/usr/bin/env python3
"""
Train PokeJAX model via Behavioral Cloning from smart heuristic.

Usage:
    # Collect data + train (default 500K transitions):
    python scripts/train_bc.py --team-pool data/team_pool.npz

    # More data:
    python scripts/train_bc.py --team-pool data/team_pool.npz --collect 1000000

    # Resume from checkpoint:
    python scripts/train_bc.py --team-pool data/team_pool.npz --resume checkpoints/bc_latest.pkl
"""

import argparse
import os
import sys
import pickle
import time

# Unbuffered output
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1, closefd=False)

import numpy as np
import jax
import jax.numpy as jnp


def main():
    parser = argparse.ArgumentParser(description="BC training for PokeJAX")
    parser.add_argument("--gen", type=int, default=4)
    parser.add_argument("--team-pool", type=str, default=None,
                        help="Path to team pool .npz file")
    parser.add_argument("--collect", type=int, default=500_000,
                        help="Number of transitions to collect")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of passes over the dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint .pkl file")
    parser.add_argument("--eval-interval", type=int, default=50,
                        help="Eval every N updates (0 to disable)")
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip all eval (avoids JIT recompilation overhead)")
    parser.add_argument("--arch", type=str, default="transformer",
                        choices=["transformer", "mlp"],
                        help="Model architecture")
    parser.add_argument("--data-file", type=str, default=None,
                        help="Load pre-collected BC data from .npz")
    parser.add_argument("--save-data", type=str, default=None,
                        help="Save collected data to .npz for reuse")
    args = parser.parse_args()

    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")

    from pokejax.env.pokejax_env import PokeJAXEnv
    from pokejax.rl.model import PokeTransformer, create_model
    from pokejax.rl.bc import (
        BCConfig, BCBatch,
        collect_bc_data,
        create_bc_train_state, make_bc_step,
        eval_vs_random, eval_vs_heuristic,
    )

    # --- Setup env ---
    env = PokeJAXEnv(gen=args.gen, team_pool_path=args.team_pool)
    print(f"Environment ready (gen {args.gen})")

    # --- Collect or load BC data ---
    if args.data_file and os.path.exists(args.data_file):
        print(f"Loading BC data from {args.data_file}...")
        data = np.load(args.data_file)
        bc_data = BCBatch(
            int_ids=data["int_ids"],
            float_feats=data["float_feats"],
            legal_mask=data["legal_mask"],
            actions=data["actions"],
        )
        print(f"  Loaded {len(bc_data.actions)} transitions")
    else:
        print(f"Collecting {args.collect} BC transitions (heuristic vs random)...")
        t0 = time.time()
        bc_data = collect_bc_data(
            env, n_transitions=args.collect, seed=args.seed,
            teacher_side=0, verbose=True,
        )
        elapsed = time.time() - t0
        print(f"  Collection took {elapsed:.1f}s ({args.collect / elapsed:.0f} trans/s)")

        if args.save_data:
            os.makedirs(os.path.dirname(args.save_data) or ".", exist_ok=True)
            np.savez_compressed(
                args.save_data,
                int_ids=bc_data.int_ids,
                float_feats=bc_data.float_feats,
                legal_mask=bc_data.legal_mask,
                actions=bc_data.actions,
            )
            print(f"  Saved BC data to {args.save_data}")

    n_data = len(bc_data.actions)

    # Action distribution
    move_count = (bc_data.actions < 4).sum()
    switch_count = (bc_data.actions >= 4).sum()
    print(f"  Actions: {move_count} moves ({100*move_count/n_data:.1f}%), "
          f"{switch_count} switches ({100*switch_count/n_data:.1f}%)")

    # --- Setup model + optimizer ---
    model = create_model(args.arch)
    print(f"Architecture: {args.arch} ({type(model).__name__})")
    cfg = BCConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        total_steps=n_data * args.epochs,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        checkpoint_dir=args.checkpoint_dir,
    )

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)

    # Resume or fresh init
    init_params = None
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        with open(args.resume, "rb") as f:
            checkpoint = pickle.load(f)
        init_params = checkpoint["params"]

    model, optimizer, train_state = create_bc_train_state(
        model, cfg, init_key, init_params=init_params,
    )
    bc_step = make_bc_step(model, optimizer)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # --- Training loop ---
    n_updates_per_epoch = n_data // cfg.batch_size
    total_updates = n_updates_per_epoch * args.epochs

    print(f"\nBC Training:")
    print(f"  Data: {n_data} transitions")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Updates/epoch: {n_updates_per_epoch}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Total updates: {total_updates}")
    print(f"  LR: {cfg.lr} -> {cfg.lr_end}")
    print()

    # Convert to JAX arrays once
    jax_int_ids = jnp.array(bc_data.int_ids)
    jax_float_feats = jnp.array(bc_data.float_feats)
    jax_legal_mask = jnp.array(bc_data.legal_mask)
    jax_actions = jnp.array(bc_data.actions, dtype=jnp.int32)

    best_win_rate = 0.0
    global_update = 0

    for epoch in range(args.epochs):
        # Shuffle data each epoch
        key, perm_key = jax.random.split(key)
        perm = np.random.RandomState(int(perm_key[0])).permutation(n_data)
        idx = jnp.array(perm)

        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0

        for batch_start in range(0, n_data - cfg.batch_size + 1, cfg.batch_size):
            batch_idx = idx[batch_start:batch_start + cfg.batch_size]

            b_int = jax_int_ids[batch_idx]
            b_float = jax_float_feats[batch_idx]
            b_mask = jax_legal_mask[batch_idx]
            b_act = jax_actions[batch_idx]

            train_state, metrics = bc_step(train_state, b_int, b_float, b_mask, b_act)

            loss_val = float(metrics["loss"])
            acc_val = float(metrics["accuracy"])
            epoch_loss += loss_val
            epoch_acc += acc_val
            n_batches += 1
            global_update += 1

            if global_update % 10 == 0:
                ent_val = float(metrics["entropy"])
                print(f"  [{global_update:>6}/{total_updates}] "
                      f"loss={loss_val:.4f} acc={acc_val:.3f} ent={ent_val:.3f}")

            # Eval
            if not args.no_eval and cfg.eval_interval > 0 and global_update % cfg.eval_interval == 0:
                print(f"\n  --- Eval at update {global_update} ---")
                t0 = time.time()

                wr_random = eval_vs_random(
                    model, train_state.params, env,
                    n_games=cfg.eval_games, seed=global_update,
                )
                print(f"  vs Random:    win={wr_random['win_rate']:.1%}  "
                      f"avg_turns={wr_random['avg_turns']:.0f}")

                wr_heur = eval_vs_heuristic(
                    model, train_state.params, env,
                    n_games=cfg.eval_games, seed=global_update + 1,
                )
                print(f"  vs Heuristic: win={wr_heur['win_rate']:.1%}  "
                      f"avg_turns={wr_heur['avg_turns']:.0f}")

                eval_time = time.time() - t0
                print(f"  Eval took {eval_time:.1f}s\n")

                # Save best checkpoint
                combined_wr = wr_random["win_rate"] * 0.5 + wr_heur["win_rate"] * 0.5
                if combined_wr > best_win_rate:
                    best_win_rate = combined_wr
                    path = os.path.join(cfg.checkpoint_dir, "bc_best.pkl")
                    with open(path, "wb") as f:
                        pickle.dump({"params": train_state.params, "step": global_update, "arch": args.arch}, f)
                    print(f"  Saved best checkpoint ({combined_wr:.1%}) -> {path}")

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_acc = epoch_acc / max(n_batches, 1)
        print(f"\n=== Epoch {epoch+1}/{args.epochs}: avg_loss={avg_loss:.4f} avg_acc={avg_acc:.3f} ===\n")

        # Save epoch checkpoint
        path = os.path.join(cfg.checkpoint_dir, "bc_latest.pkl")
        with open(path, "wb") as f:
            pickle.dump({"params": train_state.params, "step": global_update, "arch": args.arch}, f)

    # Final eval
    if not args.no_eval:
        print("\n=== Final Evaluation ===")
        wr_random = eval_vs_random(model, train_state.params, env, n_games=100, seed=0)
        print(f"vs Random (100 games):    win={wr_random['win_rate']:.1%}")
        wr_heur = eval_vs_heuristic(model, train_state.params, env, n_games=100, seed=1)
        print(f"vs Heuristic (100 games): win={wr_heur['win_rate']:.1%}")
    else:
        print("\n=== Eval skipped (--no-eval) ===")

    # Save final
    path = os.path.join(cfg.checkpoint_dir, "bc_final.pkl")
    with open(path, "wb") as f:
        pickle.dump({"params": train_state.params, "step": global_update, "arch": args.arch}, f)
    print(f"\nSaved final checkpoint -> {path}")
    print("BC training complete.")


if __name__ == "__main__":
    main()
