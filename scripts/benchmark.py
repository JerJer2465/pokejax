"""
PokeJAX throughput benchmark.

Measures end-to-end rollout speed (steps/second) after JIT/XLA compilation.
Reports:
  - Single-env JIT step throughput
  - Batched vmap throughput (B=64, 256, 1024)
  - lax.scan rollout throughput (N-step trajectory)
  - vmap + lax.scan combined

Usage:
    py scripts/benchmark.py [--gen 4] [--batch 256] [--steps 100] [--warmup 3] [--reps 10]

All numbers are for STEADY-STATE throughput (after the first JIT compilation).
Compilation time is reported separately per benchmark.
"""

import argparse
import time
import sys
import os

import jax
import jax.numpy as jnp
import numpy as np

# Enable persistent compilation cache — first run compiles (~10 min), all future runs are instant.
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_compile_cache")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gen",    type=int, default=4,    help="Generation (default 4)")
    p.add_argument("--batch",  type=int, default=64,   help="vmap batch size (default 64)")
    p.add_argument("--steps",  type=int, default=100,  help="lax.scan rollout length")
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--warmup", type=int, default=2,    help="Warmup calls before timing")
    p.add_argument("--reps",   type=int, default=5,    help="Timed repetitions")
    return p.parse_args()


# ---------------------------------------------------------------------------
# State factory
# ---------------------------------------------------------------------------

def make_state(tables, cfg, key):
    """Create a minimal BattleState suitable for benchmarking."""
    from pokejax.core.state import make_battle_state

    n = 6
    base       = np.array([[80, 80, 80, 80, 80, 80]] * n, dtype=np.int16)
    max_hp_arr = np.full(n, 250, dtype=np.int16)
    types      = np.zeros((n, 2), dtype=np.int8)
    types[:, 0] = 1   # Normal
    move_ids   = np.tile(np.arange(4, dtype=np.int16), (n, 1))
    move_pp    = np.full((n, 4), 35, dtype=np.int8)
    levels     = np.full(n, 100, dtype=np.int8)
    zeros_n    = np.zeros(n, dtype=np.int16)
    zeros_i8   = np.zeros(n, dtype=np.int8)

    return make_battle_state(
        p1_species=zeros_n, p2_species=zeros_n,
        p1_abilities=zeros_n, p2_abilities=zeros_n,
        p1_items=zeros_n, p2_items=zeros_n,
        p1_types=types, p2_types=types,
        p1_base_stats=base, p2_base_stats=base,
        p1_max_hp=max_hp_arr, p2_max_hp=max_hp_arr,
        p1_move_ids=move_ids, p2_move_ids=move_ids,
        p1_move_pp=move_pp, p2_move_pp=move_pp,
        p1_move_max_pp=move_pp, p2_move_max_pp=move_pp,
        p1_levels=levels, p2_levels=levels,
        p1_genders=zeros_i8, p2_genders=zeros_i8,
        p1_natures=zeros_i8, p2_natures=zeros_i8,
        p1_weights_hg=np.full(n, 100, dtype=np.int16),
        p2_weights_hg=np.full(n, 100, dtype=np.int16),
        rng_key=key,
    )


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def measure(fn, *args, warmup: int = 2, reps: int = 5, label: str = ""):
    """
    Time fn(*args): warmup calls (includes first JIT), then reps timed calls.
    Returns (compile_time_s, mean_step_s, std_step_s).
    """
    # Warmup
    t0 = time.perf_counter()
    for _ in range(warmup):
        out = fn(*args)
        jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0

    # Timed reps
    elapsed = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        elapsed.append(time.perf_counter() - t0)

    mean_s = float(np.mean(elapsed))
    std_s  = float(np.std(elapsed))

    compile_ms = compile_s * 1e3
    mean_ms    = mean_s * 1e3
    std_ms     = std_s * 1e3
    print(f"    {label:<45s}  {mean_ms:8.2f} ms ± {std_ms:.2f} ms   "
          f"[compile: {compile_ms:.0f} ms]")
    return compile_s, mean_s, std_s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"\nPokeJAX Throughput Benchmark")
    print(f"  Backend : {jax.default_backend().upper()} ({len(jax.devices())} device(s))")
    print(f"  JAX     : {jax.__version__}")
    print(f"  Gen     : {args.gen}  |  batch={args.batch}  |  steps={args.steps}")
    print()

    # ------------------------------------------------------------------ setup
    print("Loading tables and env…")
    from pokejax.data.tables import load_tables
    from pokejax.config import GenConfig
    from pokejax.env.pokejax_env import PokeJAXEnv

    env    = PokeJAXEnv(gen=args.gen)
    tables = env.tables
    cfg    = env.cfg
    print("  done.\n")

    key = jax.random.PRNGKey(args.seed)

    # Single state
    key, sk = jax.random.split(key)
    state   = make_state(tables, cfg, sk)
    actions = jnp.zeros(2, dtype=jnp.int32)

    # Batch of states
    key, sk = jax.random.split(key)
    batch_keys = jax.random.split(sk, args.batch)
    print(f"  Building batch of {args.batch} states…", end=" ", flush=True)
    batch_states  = jax.vmap(lambda k: make_state(tables, cfg, k))(batch_keys)
    batch_actions = jnp.zeros((args.batch, 2), dtype=jnp.int32)
    print("done.\n")

    # ------------------------------------------------------------------ benchmarks
    results = {}

    # 1. Single-env JIT step
    print("1. Single-env JIT step")
    key, sk = jax.random.split(key)
    step_jit = jax.jit(lambda s, a, k: env.step(s, a, k))
    _, mean_s, _ = measure(step_jit, state, actions, sk,
                            warmup=args.warmup, reps=args.reps,
                            label="step_jit (1 env × 1 step)")
    results["single_jit"] = 1.0 / mean_s
    print(f"      → {results['single_jit']:>12,.0f} steps/sec\n")

    # 2. vmap batch step
    print(f"2. Batched vmap step  (B={args.batch})")
    key, sk = jax.random.split(key)
    batch_step_keys = jax.random.split(sk, args.batch)
    step_vmap = jax.jit(jax.vmap(lambda s, a, k: env.step(s, a, k)))
    _, mean_vmap, _ = measure(step_vmap, batch_states, batch_actions, batch_step_keys,
                               warmup=args.warmup, reps=args.reps,
                               label=f"step_vmap (B={args.batch} × 1 step)")
    results["vmap_step"] = args.batch / mean_vmap
    print(f"      → {results['vmap_step']:>12,.0f} steps/sec\n")

    # 3. lax.scan rollout (single env)
    print(f"3. lax.scan rollout  (1 env × {args.steps} steps)")
    key, sk = jax.random.split(key)

    @jax.jit
    def rollout_single(init_state, init_key):
        def _step(carry, _):
            s, k = carry
            k, sk = jax.random.split(k)
            acts = jnp.zeros(2, dtype=jnp.int32)
            next_s, _, _, _, _ = env.step(s, acts, sk)
            return (next_s, k), next_s.turn
        (final_s, _), turns = jax.lax.scan(
            _step, (init_state, init_key), None, length=args.steps
        )
        return final_s, turns

    _, mean_scan, _ = measure(rollout_single, state, sk,
                               warmup=args.warmup, reps=args.reps,
                               label=f"lax.scan (1 × {args.steps} steps)")
    results["scan"] = args.steps / mean_scan
    print(f"      → {results['scan']:>12,.0f} steps/sec\n")

    # 4. vmap + lax.scan (batched rollout)
    print(f"4. vmap + lax.scan   (B={args.batch} × {args.steps} steps)")
    key, sk = jax.random.split(key)
    batch_rollout_keys = jax.random.split(sk, args.batch)

    @jax.jit
    def rollout_batched(init_states, init_keys):
        def _one_rollout(s0, k0):
            def _step(carry, _):
                s, k = carry
                k, sk = jax.random.split(k)
                acts = jnp.zeros(2, dtype=jnp.int32)
                next_s, _, _, _, _ = env.step(s, acts, sk)
                return (next_s, k), next_s.turn
            (final_s, _), turns = jax.lax.scan(
                _step, (s0, k0), None, length=args.steps
            )
            return final_s, turns
        return jax.vmap(_one_rollout)(init_states, init_keys)

    _, mean_vmap_scan, _ = measure(
        rollout_batched, batch_states, batch_rollout_keys,
        warmup=args.warmup, reps=args.reps,
        label=f"vmap+scan (B={args.batch} × {args.steps})"
    )
    results["vmap_scan"] = (args.batch * args.steps) / mean_vmap_scan
    print(f"      → {results['vmap_scan']:>12,.0f} steps/sec\n")

    # ------------------------------------------------------------------ summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  single step JIT         : {results['single_jit']:>12,.0f}  steps/sec")
    print(f"  vmap step (B={args.batch:<4d})      : {results['vmap_step']:>12,.0f}  steps/sec")
    print(f"  lax.scan ({args.steps} steps)    : {results['scan']:>12,.0f}  steps/sec")
    print(f"  vmap+scan (B={args.batch:<4d})      : {results['vmap_scan']:>12,.0f}  steps/sec")
    print()
    print(f"  GPU RL training target  :    1,000,000+  steps/sec")
    print()


if __name__ == "__main__":
    main()
