#!/usr/bin/env python3
"""
Play pokejax model on the public Pokemon Showdown ladder using expectimax
search for move selection. Uses JAX CPU on Windows for search.

Pre-compiles all JIT kernels before connecting to PS to avoid websocket
timeouts during compilation.

Usage:
    cd /c/Users/jerry/Documents/Coding/pokejax && /c/Windows/py.exe -3 scripts/play_ladder_mcts.py --checkpoint checkpoints/BC-250M/ppo_best.pkl --games 100

    # Use gen9 for faster matchmaking during testing:
    cd /c/Users/jerry/Documents/Coding/pokejax && /c/Windows/py.exe -3 scripts/play_ladder_mcts.py --checkpoint checkpoints/BC-250M/ppo_best.pkl --games 5 --format gen9randombattle
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.stdout = open(sys.stdout.fileno(), "w", buffering=1, closefd=False)

# Force CPU backend to avoid XLA AOT warnings on Windows
os.environ["JAX_PLATFORMS"] = "cpu"

# Remove proxy env vars so websockets connects directly
for k in list(os.environ):
    if k.lower() in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        del os.environ[k]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def main():
    parser = argparse.ArgumentParser(
        description="Play pokejax on PS ladder with expectimax search"
    )
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/BC-250M/ppo_best.pkl")
    parser.add_argument("--gen", type=int, default=4)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--format", type=str, default="gen4randombattle")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-dir", type=str, default="ladder_results_mcts")
    parser.add_argument("--username", type=str, default="paganini2465")
    parser.add_argument("--password", type=str, default="20040605jY")

    # Search parameters
    parser.add_argument("--n-samples", type=int, default=16,
                        help="RNG samples per action pair (default: 16)")
    parser.add_argument("--opp-temperature", type=float, default=0.5,
                        help="Opponent policy temperature (default: 0.5)")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    fh = logging.FileHandler(out_dir / "ladder_mcts.log")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    log = logging.getLogger("ladder-mcts")

    log.info("Loading model from %s", args.checkpoint)
    log.info("Search config: n_samples=%d, opp_temp=%.2f",
             args.n_samples, args.opp_temperature)
    log.info("User: %s | Format: %s | Games: %d",
             args.username, args.format, args.games)

    # ── Pre-compile ALL JIT kernels BEFORE connecting to PS ──
    # Compilation can take 30-120s on CPU, which would block the event loop
    # and cause websocket timeouts. So we compile everything synchronously first.
    import pickle
    import jax
    import jax.numpy as jnp
    from pokejax.rl.model import create_model
    from pokejax.data.tables import load_tables
    from pokejax.env.pokejax_env import PokeJAXEnv
    from pokejax.search.expectimax import ExpectiMaxSearch
    from pokejax.players.showdown_player import N_TOKENS, FLOAT_DIM, N_ACTIONS, INT_IDS_PER_TOKEN

    log.info("Device: %s", jax.devices()[0])
    log.info("Pre-compiling search kernels (this may take a few minutes on CPU)...")

    tables = load_tables(args.gen)
    with open(args.checkpoint, "rb") as f:
        ckpt = pickle.load(f)
    arch = ckpt.get("arch", "transformer")
    model = create_model(arch)
    params = ckpt["params"]
    env = PokeJAXEnv(gen=args.gen)

    # Compile + warmup search kernels
    searcher = ExpectiMaxSearch(
        env=env, model=model, params=params,
        n_samples=args.n_samples, opp_temperature=args.opp_temperature,
        warmup=True,  # Forces JIT compilation now
    )

    # Also compile the fallback forward pass
    @jax.jit
    def _fwd(params, int_ids, float_feats, legal_mask):
        log_probs, _, value = model.apply(
            params, int_ids[None], float_feats[None], legal_mask[None],
        )
        return log_probs[0], value[0]

    dummy_int = jnp.zeros((N_TOKENS, INT_IDS_PER_TOKEN), dtype=jnp.int32)
    dummy_float = jnp.zeros((N_TOKENS, FLOAT_DIM), dtype=jnp.float32)
    dummy_mask = jnp.ones(N_ACTIONS, dtype=jnp.float32)
    _ = _fwd(params, dummy_int, dummy_float, dummy_mask)

    log.info("All kernels compiled! Now connecting to PS...")

    # ── Connect to PS using standard poke-env ──
    # Monkey-patch websockets to disable proxy detection.
    # websockets v15+ auto-detects system proxy (Clash TUN), causing timeouts.
    import websockets
    import websockets.asyncio.client as _wsc
    _orig_connect = _wsc.connect
    class _no_proxy_connect(_orig_connect):
        def __init__(self, *a, **kw):
            kw.setdefault("proxy", None)
            super().__init__(*a, **kw)
    # Patch everywhere websockets.connect might be looked up
    _wsc.connect = _no_proxy_connect
    websockets.connect = _no_proxy_connect

    from poke_env import ShowdownServerConfiguration, AccountConfiguration
    # Also patch the already-imported ws reference in poke-env
    import poke_env.ps_client.ps_client as _ps_mod
    _ps_mod.ws.connect = _no_proxy_connect

    from pokejax.players.mcts_player import MctsPlayer

    player = MctsPlayer(
        checkpoint_path=args.checkpoint,
        gen=args.gen,
        n_samples=args.n_samples,
        opp_temperature=args.opp_temperature,
        verbose=args.verbose,
        prebuilt_searcher=searcher,
        prebuilt_params=params,
        account_configuration=AccountConfiguration(args.username, args.password),
        server_configuration=ShowdownServerConfiguration,
        battle_format=args.format,
    )
    log.info("Player created, waiting for PS connection...")

    # Wait for login
    t_login = time.time()
    while not player.ps_client.logged_in.is_set():
        await asyncio.sleep(0.5)
        if time.time() - t_login > 60:
            log.error("Failed to connect/login after 60s")
            return
    log.info("Logged in! Starting ladder...")

    results_file = out_dir / "ladder_results.jsonl"
    elo_history = []
    wins, losses, ties = 0, 0, 0
    seen_battles = set()
    start = time.time()
    total = 0

    while total < args.games:
        try:
            await player.ladder(1)
        except Exception as e:
            log.error("Ladder error: %s", e, exc_info=True)
            await asyncio.sleep(5)
            continue

        for tag, battle in player.battles.items():
            if tag in seen_battles or not battle.finished:
                continue
            seen_battles.add(tag)

            won = battle.won
            if won is True:
                wins += 1
                result = "WIN"
            elif won is False:
                losses += 1
                result = "LOSS"
            else:
                ties += 1
                result = "TIE"

            total += 1
            rating = battle.rating
            record = {
                "game": total, "battle_tag": tag, "result": result,
                "turns": battle.turn, "rating": rating,
                "opponent_rating": battle.opponent_rating,
                "wins": wins, "losses": losses, "ties": ties,
                "win_rate": round(wins / max(total, 1), 4),
                "elapsed_s": round(time.time() - start, 1),
                "search_samples": args.n_samples,
            }
            elo_history.append(record)

            with open(results_file, "a") as f:
                f.write(json.dumps(record) + "\n")

            elo_str = f"Elo {rating}" if rating else "Elo ?"
            log.info("[%d/%d] %s in %dt | %s | %dW/%dL/%dT (%.0f%%)",
                     total, args.games, result, battle.turn, elo_str,
                     wins, losses, ties, wins / total * 100)

        if total % 10 == 0 and elo_history:
            _plot_elo(elo_history, out_dir)

    # Final summary
    total = wins + losses + ties
    log.info("=" * 60)
    log.info("DONE: %dW/%dL/%dT (%.1f%%) in %.0f minutes",
             wins, losses, ties, wins / max(total, 1) * 100,
             (time.time() - start) / 60)
    if elo_history and elo_history[-1]["rating"]:
        log.info("Final Elo: %s", elo_history[-1]["rating"])
    log.info("=" * 60)

    with open(out_dir / "ladder_summary.json", "w") as f:
        json.dump({
            "checkpoint": args.checkpoint, "format": args.format,
            "total_games": total, "wins": wins, "losses": losses, "ties": ties,
            "win_rate": round(wins / max(total, 1), 4),
            "final_elo": elo_history[-1]["rating"] if elo_history else None,
            "search_samples": args.n_samples,
            "opp_temperature": args.opp_temperature,
            "history": elo_history,
        }, f, indent=2)

    _plot_elo(elo_history, out_dir)


def _plot_elo(history, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    rated = [(r["game"], r["rating"]) for r in history if r.get("rating")]
    if not rated:
        return

    games, ratings = zip(*rated)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                    gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(games, ratings, "b-", linewidth=1.5, alpha=0.8)
    ax1.scatter(games, ratings, c="blue", s=8, alpha=0.4, zorder=3)
    ax1.set_xlabel("Game Number")
    ax1.set_ylabel("Elo Rating")
    ax1.set_title(f"PS Ladder (MCTS) - Gen 4 Random Battle\n"
                  f"Latest: {ratings[-1]} Elo after {games[-1]} games")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1000, color="gray", linestyle="--", alpha=0.5, label="Start (1000)")
    if len(ratings) > 1:
        ax1.axhline(y=max(ratings), color="green", linestyle=":", alpha=0.4,
                     label=f"Peak: {max(ratings)}")
    ax1.legend()

    w = 0
    wr = []
    for r in history:
        if r["result"] == "WIN":
            w += 1
        wr.append(w / r["game"] * 100)
    ax2.plot([r["game"] for r in history], wr, "g-", linewidth=1.5)
    ax2.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Game Number")
    ax2.set_ylabel("Cumulative Win Rate (%)")
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "elo_graph.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    asyncio.run(main())
