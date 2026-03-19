#!/usr/bin/env python3
"""
Train PokeJAX model via Behavioral Cloning from smart heuristic.

Uses fully vectorized data collection (vmap + lax.scan) for GPU speed.
Supports heuristic-vs-random and heuristic-vs-heuristic data collection.

Usage:
    # Default: 2M transitions from heuristic vs random, transformer:
    python scripts/train_bc.py

    # Heuristic vs heuristic (higher quality data):
    python scripts/train_bc.py --opp-mode heuristic

    # Mixed data (both random and heuristic opponents):
    python scripts/train_bc.py --opp-mode mixed

    # More data + more epochs:
    python scripts/train_bc.py --n-envs 1024 --n-steps 512 --epochs 20

    # Resume from checkpoint:
    python scripts/train_bc.py --resume checkpoints/bc_latest.pkl

    # TensorBoard:
    tensorboard --logdir runs --host 0.0.0.0
"""

import argparse
import os
import sys
import pickle
import time
from datetime import datetime
from pathlib import Path

# Unbuffered output
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1, closefd=False)

# Persistent XLA compilation cache — avoids 3+ minute recompilation
_CACHE_DIR = str(Path(__file__).resolve().parent.parent / ".jax_cache")
import jax
jax.config.update("jax_compilation_cache_dir", _CACHE_DIR)

import numpy as np
import jax.numpy as jnp


def main():
    parser = argparse.ArgumentParser(description="BC training for PokeJAX")
    parser.add_argument("--gen", type=int, default=4)
    parser.add_argument("--team-pool", type=str, default=None,
                        help="Path to team pool .npz file")

    # Data collection
    parser.add_argument("--n-envs", type=int, default=512,
                        help="Number of parallel envs for data collection")
    parser.add_argument("--n-steps", type=int, default=256,
                        help="Steps per env for data collection")
    parser.add_argument("--opp-mode", type=str, default="mixed",
                        choices=["random", "heuristic", "mixed"],
                        help="Opponent policy for data collection")
    parser.add_argument("--collect", type=int, default=None,
                        help="Override: use legacy sequential collection with N transitions")

    # Training
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-end", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of passes over the dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint .pkl file")

    # Eval
    parser.add_argument("--eval-interval", type=int, default=100,
                        help="Eval every N updates (0 to disable)")
    parser.add_argument("--eval-games", type=int, default=64)
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip all eval")

    # Model
    parser.add_argument("--arch", type=str, default="transformer",
                        choices=["transformer", "mlp"],
                        help="Model architecture")

    # Data save/load
    parser.add_argument("--data-file", type=str, default=None,
                        help="Load pre-collected BC data from .npz")
    parser.add_argument("--save-data", type=str, default=None,
                        help="Save collected data to .npz for reuse")

    # TensorBoard
    parser.add_argument("--tensorboard-dir", type=str, default="runs",
                        help="TensorBoard log directory")
    parser.add_argument("--no-tensorboard", action="store_true",
                        help="Disable TensorBoard logging")

    args = parser.parse_args()

    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")

    from pokejax.env.pokejax_env import PokeJAXEnv
    from pokejax.rl.model import create_model
    from pokejax.rl.bc import (
        BCConfig, BCBatch,
        collect_bc_data, collect_bc_data_vectorized,
        create_bc_train_state, make_bc_step,
        eval_vs_random, eval_vs_heuristic,
    )

    # --- Setup env ---
    env = PokeJAXEnv(gen=args.gen, team_pool_path=args.team_pool)
    print(f"Environment ready (gen {args.gen})")

    # --- TensorBoard ---
    writer = None
    if not args.no_tensorboard:
        try:
            from tensorboardX import SummaryWriter
            run_name = f"bc_{args.arch}_{args.opp_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(args.tensorboard_dir, exist_ok=True)
            writer = SummaryWriter(os.path.join(args.tensorboard_dir, run_name))
            print(f"TensorBoard: {args.tensorboard_dir}/{run_name}")
        except ImportError:
            print("  tensorboardX not installed, skipping TensorBoard logging")

    # --- Collect or load BC data ---
    # Auto-detect saved data: check --data-file, then --save-data path, then default
    default_data_path = os.path.join("data", "bc_mixed.npz")
    data_load_path = args.data_file or args.save_data or default_data_path

    if data_load_path and os.path.exists(data_load_path) and args.collect is None:
        print(f"Loading BC data from {data_load_path}...")
        data = np.load(data_load_path)
        bc_data = BCBatch(
            int_ids=data["int_ids"],
            float_feats=data["float_feats"],
            legal_mask=data["legal_mask"],
            actions=data["actions"],
        )
        print(f"  Loaded {len(bc_data.actions)} transitions")
    elif args.collect is not None:
        # Legacy sequential collection
        print(f"Collecting {args.collect} BC transitions (sequential, heuristic vs random)...")
        t0 = time.time()
        bc_data = collect_bc_data(
            env, n_transitions=args.collect, seed=args.seed,
            teacher_side=0, verbose=True,
        )
        elapsed = time.time() - t0
        print(f"  Collection took {elapsed:.1f}s ({args.collect / elapsed:.0f} trans/s)")
    else:
        # Vectorized collection (default)
        if args.opp_mode == "mixed":
            # Collect half vs random, half vs heuristic
            half_steps = args.n_steps // 2
            print(f"\n=== Collecting BC data (mixed opponents) ===")

            print(f"\n--- Phase 1: heuristic vs random ({args.n_envs} × {half_steps}) ---")
            t0 = time.time()
            bc_random = collect_bc_data_vectorized(
                env, n_envs=args.n_envs, n_steps=half_steps,
                seed=args.seed, teacher_side=0, opp_mode="random",
                verbose=True,
            )
            print(f"  Phase 1: {len(bc_random.actions)} transitions in {time.time()-t0:.1f}s")

            print(f"\n--- Phase 2: heuristic vs heuristic ({args.n_envs} × {half_steps}) ---")
            t0 = time.time()
            bc_heur = collect_bc_data_vectorized(
                env, n_envs=args.n_envs, n_steps=half_steps,
                seed=args.seed + 1000, teacher_side=0, opp_mode="heuristic",
                verbose=True,
            )
            print(f"  Phase 2: {len(bc_heur.actions)} transitions in {time.time()-t0:.1f}s")

            # Merge
            bc_data = BCBatch(
                int_ids=np.concatenate([bc_random.int_ids, bc_heur.int_ids]),
                float_feats=np.concatenate([bc_random.float_feats, bc_heur.float_feats]),
                legal_mask=np.concatenate([bc_random.legal_mask, bc_heur.legal_mask]),
                actions=np.concatenate([bc_random.actions, bc_heur.actions]),
            )
            print(f"\n  Total: {len(bc_data.actions)} transitions (mixed)")
        else:
            print(f"\nCollecting BC data ({args.n_envs} envs × {args.n_steps} steps, "
                  f"opp={args.opp_mode})...")
            t0 = time.time()
            bc_data = collect_bc_data_vectorized(
                env, n_envs=args.n_envs, n_steps=args.n_steps,
                seed=args.seed, teacher_side=0, opp_mode=args.opp_mode,
                verbose=True,
            )
            elapsed = time.time() - t0
            print(f"  Collection took {elapsed:.1f}s")

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

    # Action distribution analysis
    move_count = (bc_data.actions < 4).sum()
    switch_count = (bc_data.actions >= 4).sum()
    print(f"\n  Actions: {move_count} moves ({100*move_count/n_data:.1f}%), "
          f"{switch_count} switches ({100*switch_count/n_data:.1f}%)")

    # Per-action breakdown
    for i in range(10):
        count = (bc_data.actions == i).sum()
        label = f"move_{i}" if i < 4 else f"switch_{i-4}"
        print(f"    {label}: {count} ({100*count/n_data:.1f}%)")

    # --- Setup model + optimizer ---
    model = create_model(args.arch)
    print(f"\nArchitecture: {args.arch} ({type(model).__name__})")
    cfg = BCConfig(
        lr=args.lr,
        lr_end=args.lr_end,
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

    # Count params
    n_params = sum(x.size for x in jax.tree.leaves(train_state.params))
    print(f"  Parameters: {n_params:,}")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # --- Training loop ---
    n_updates_per_epoch = n_data // cfg.batch_size
    total_updates = n_updates_per_epoch * args.epochs

    print(f"\nBC Training:")
    print(f"  Data: {n_data:,} transitions")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Updates/epoch: {n_updates_per_epoch}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Total updates: {total_updates}")
    print(f"  LR: {cfg.lr} -> {cfg.lr_end}")
    print()

    # Keep data on CPU (numpy) — transfer only per-batch to GPU.
    # This avoids GPU OOM with large datasets (262K × 15 × 394 ≈ 6GB).
    np_int_ids = np.asarray(bc_data.int_ids)
    np_float_feats = np.asarray(bc_data.float_feats)
    np_legal_mask = np.asarray(bc_data.legal_mask)
    np_actions = np.asarray(bc_data.actions, dtype=np.int32)

    best_win_rate = 0.0
    global_update = 0
    train_start = time.time()

    for epoch in range(args.epochs):
        # Shuffle data each epoch
        key, perm_key = jax.random.split(key)
        perm = np.random.RandomState(int(perm_key[0])).permutation(n_data)

        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0

        for batch_start in range(0, n_data - cfg.batch_size + 1, cfg.batch_size):
            batch_idx = perm[batch_start:batch_start + cfg.batch_size]

            # Transfer only this batch to GPU
            b_int = jnp.array(np_int_ids[batch_idx])
            b_float = jnp.array(np_float_feats[batch_idx])
            b_mask = jnp.array(np_legal_mask[batch_idx])
            b_act = jnp.array(np_actions[batch_idx])

            train_state, metrics = bc_step(train_state, b_int, b_float, b_mask, b_act)

            loss_val = float(metrics["loss"])
            acc_val = float(metrics["accuracy"])
            ent_val = float(metrics["entropy"])
            epoch_loss += loss_val
            epoch_acc += acc_val
            n_batches += 1
            global_update += 1

            # TensorBoard logging
            if writer is not None:
                writer.add_scalar("bc/loss", loss_val, global_update)
                writer.add_scalar("bc/accuracy", acc_val, global_update)
                writer.add_scalar("bc/entropy", ent_val, global_update)

            if global_update % 20 == 0:
                elapsed = time.time() - train_start
                sps = global_update * cfg.batch_size / max(elapsed, 0.001)
                print(f"  [{global_update:>6}/{total_updates}] "
                      f"loss={loss_val:.4f} acc={acc_val:.3f} ent={ent_val:.3f} "
                      f"({sps:.0f} samples/s)")

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

                if writer is not None:
                    writer.add_scalar("eval/win_rate_vs_random", wr_random["win_rate"], global_update)
                    writer.add_scalar("eval/win_rate_vs_heuristic", wr_heur["win_rate"], global_update)
                    writer.add_scalar("eval/avg_turns_vs_random", wr_random["avg_turns"], global_update)
                    writer.add_scalar("eval/avg_turns_vs_heuristic", wr_heur["avg_turns"], global_update)

                # Save best checkpoint
                combined_wr = wr_random["win_rate"] * 0.3 + wr_heur["win_rate"] * 0.7
                if combined_wr > best_win_rate:
                    best_win_rate = combined_wr
                    path = os.path.join(cfg.checkpoint_dir, "bc_best.pkl")
                    with open(path, "wb") as f:
                        pickle.dump({
                            "params": train_state.params,
                            "step": global_update,
                            "arch": args.arch,
                        }, f)
                    print(f"  Saved best checkpoint (combined={combined_wr:.1%}) -> {path}")

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_acc = epoch_acc / max(n_batches, 1)
        print(f"\n=== Epoch {epoch+1}/{args.epochs}: avg_loss={avg_loss:.4f} avg_acc={avg_acc:.3f} ===\n")

        if writer is not None:
            writer.add_scalar("bc/epoch_loss", avg_loss, epoch + 1)
            writer.add_scalar("bc/epoch_accuracy", avg_acc, epoch + 1)

        # Save epoch checkpoint
        path = os.path.join(cfg.checkpoint_dir, "bc_latest.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "params": train_state.params,
                "step": global_update,
                "arch": args.arch,
            }, f)

    # Final eval
    if not args.no_eval:
        print("\n=== Final Evaluation (200 games each) ===")
        wr_random = eval_vs_random(model, train_state.params, env, n_games=200, seed=0)
        print(f"vs Random (200 games):    win={wr_random['win_rate']:.1%}  "
              f"avg_turns={wr_random['avg_turns']:.0f}")
        wr_heur = eval_vs_heuristic(model, train_state.params, env, n_games=200, seed=1)
        print(f"vs Heuristic (200 games): win={wr_heur['win_rate']:.1%}  "
              f"avg_turns={wr_heur['avg_turns']:.0f}")

        if writer is not None:
            writer.add_scalar("eval/final_win_rate_vs_random", wr_random["win_rate"], global_update)
            writer.add_scalar("eval/final_win_rate_vs_heuristic", wr_heur["win_rate"], global_update)
    else:
        print("\n=== Eval skipped (--no-eval) ===")

    # Save final
    path = os.path.join(cfg.checkpoint_dir, "bc_final.pkl")
    with open(path, "wb") as f:
        pickle.dump({
            "params": train_state.params,
            "step": global_update,
            "arch": args.arch,
        }, f)
    print(f"\nSaved final checkpoint -> {path}")

    total_time = time.time() - train_start
    print(f"BC training complete in {total_time:.0f}s ({total_time/60:.1f}m)")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
