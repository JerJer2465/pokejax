#!/usr/bin/env python3
"""
Play pokejax model on Pokemon Showdown ladder using true MCTS tree search.

This uses GPU-accelerated Monte Carlo Tree Search (AlphaZero-style PUCT)
for multi-turn lookahead, unlike the depth-1 expectimax in play_ladder_mcts.py.
Designed to run in WSL with CUDA for maximum search power.

The MCTS search:
  - Builds a search tree via PUCT selection + neural network priors
  - Simulates future turns using the pokejax GPU engine
  - Evaluates leaf positions with the value network
  - Handles opponent by sampling from the policy network
  - Handles stochasticity via RNG sampling (damage rolls, accuracy, etc.)

Usage (from WSL):
    cd /mnt/c/Users/jerry/Documents/Coding/pokejax && \
    python3 scripts/play_ladder_tree_mcts.py \
        --checkpoint checkpoints/BC-250M/ppo_best.pkl \
        --games 100 --simulations 128

Usage (from Windows shell, launching WSL):
    cd /c/Users/jerry/Documents/Coding/pokejax && wsl bash -c \
        "cd /mnt/c/Users/jerry/Documents/Coding/pokejax && \
         python3 scripts/play_ladder_tree_mcts.py \
         --checkpoint checkpoints/BC-250M/ppo_best.pkl \
         --games 100 --simulations 128"

Networking:
    WSL needs to reach play.pokemonshowdown.com. Options:
    1. WSL2 mirrored networking mode (networkingMode=mirrored in .wslconfig)
    2. Forward host proxy: set --server-url wss://sim3.psim.us/showdown/websocket
    3. Use the ws_proxy.py bridge: run ws_proxy.py on Windows, connect from WSL
       via --server-url ws://localhost:8088

    For the ws_proxy approach (recommended):
        # Terminal 1 (Windows): python scripts/ws_proxy.py
        # Terminal 2 (WSL):     python3 scripts/play_ladder_tree_mcts.py --local-proxy
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import pickle
from pathlib import Path

sys.stdout = open(sys.stdout.fileno(), "w", buffering=1, closefd=False)

# Force CPU backend on Windows (CUDA only available in WSL)
if sys.platform == "win32":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Clear proxy env vars so websockets connects directly
# (Clash TUN + DIRECT rules handle routing at the network level)
for k in list(os.environ):
    if k.lower() in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        del os.environ[k]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def main():
    parser = argparse.ArgumentParser(
        description="Play pokejax on PS ladder with MCTS tree search (GPU)"
    )
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/BC-250M/ppo_best.pkl")
    parser.add_argument("--gen", type=int, default=4)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--format", type=str, default="gen4randombattle")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-dir", type=str, default="ladder_results_tree_mcts")
    parser.add_argument("--username", type=str, default="paganini2465")
    parser.add_argument("--password", type=str, default="20040605jY")

    # MCTS parameters
    parser.add_argument("--simulations", type=int, default=128,
                        help="MCTS simulations per move (default: 128)")
    parser.add_argument("--c-puct", type=float, default=2.5,
                        help="PUCT exploration constant (default: 2.5)")
    parser.add_argument("--opp-temperature", type=float, default=0.5,
                        help="Opponent policy temperature (default: 0.5)")
    parser.add_argument("--max-depth", type=int, default=10,
                        help="Max search depth in turns (default: 10)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for GPU leaf expansion (default: 8)")
    parser.add_argument("--batched", action="store_true",
                        help="Use virtual-loss batched search (default: sequential)")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3,
                        help="Dirichlet noise alpha (default: 0.3)")
    parser.add_argument("--dirichlet-frac", type=float, default=0.25,
                        help="Dirichlet noise fraction (default: 0.25)")

    # Connection
    parser.add_argument("--local-proxy", action="store_true",
                        help="Connect via local ws_proxy.py (ws://localhost:8088)")
    parser.add_argument("--server-url", type=str, default=None,
                        help="Override PS server URL")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root_logger.addHandler(sh)
    fh = logging.FileHandler(out_dir / "ladder_tree_mcts.log")
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)
    log = logging.getLogger("tree-mcts")

    # ── Pre-compile JIT kernels BEFORE connecting to PS ──
    # MCTS kernel compilation can take 30-120s; do it before websocket connect
    # to avoid timeouts.
    import jax
    import jax.numpy as jnp

    devices = jax.devices()
    log.info("JAX devices: %s", devices)
    log.info("Loading model from %s", args.checkpoint)
    log.info("MCTS config: sims=%d, c_puct=%.2f, opp_temp=%.2f, "
             "depth=%d, batch=%d, batched=%s",
             args.simulations, args.c_puct, args.opp_temperature,
             args.max_depth, args.batch_size, args.batched)

    from pokejax.rl.model import create_model
    from pokejax.data.tables import load_tables
    from pokejax.env.pokejax_env import PokeJAXEnv
    from pokejax.search.mcts import MCTSSearch

    log.info("Pre-compiling MCTS kernels (this may take a few minutes)...")
    t_compile = time.time()

    tables = load_tables(args.gen)
    with open(args.checkpoint, "rb") as f:
        ckpt = pickle.load(f)
    arch = ckpt.get("arch", "transformer")
    model = create_model(arch)
    params = ckpt["params"]
    env = PokeJAXEnv(gen=args.gen)

    # Build + warmup MCTS
    searcher = MCTSSearch(
        env=env, model=model, params=params,
        n_simulations=args.simulations,
        c_puct=args.c_puct,
        opp_temperature=args.opp_temperature,
        max_depth=args.max_depth,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_frac=args.dirichlet_frac,
        batch_size=args.batch_size,
        warmup=True,
    )

    log.info("MCTS kernels compiled in %.1fs", time.time() - t_compile)

    # ── Connect to PS ──
    from poke_env import ShowdownServerConfiguration, AccountConfiguration

    # Determine server config
    PS_AUTH_URL = "https://play.pokemonshowdown.com/action.php?"

    if args.local_proxy:
        from poke_env import ServerConfiguration
        server_config = ServerConfiguration(
            "localhost:8088", PS_AUTH_URL,
        )
        log.info("Connecting via local proxy: localhost:8088")
    elif args.server_url:
        from poke_env import ServerConfiguration
        server_config = ServerConfiguration(
            args.server_url, PS_AUTH_URL,
        )
        log.info("Connecting to: %s", args.server_url)
    else:
        server_config = ShowdownServerConfiguration
        log.info("Connecting to: play.pokemonshowdown.com")

    from pokejax.players.tree_search_player import TreeSearchPlayer

    player = TreeSearchPlayer(
        checkpoint_path=args.checkpoint,
        gen=args.gen,
        n_simulations=args.simulations,
        c_puct=args.c_puct,
        opp_temperature=args.opp_temperature,
        max_depth=args.max_depth,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_frac=args.dirichlet_frac,
        batch_size=args.batch_size,
        use_batched=args.batched,
        verbose=args.verbose,
        prebuilt_searcher=searcher,
        prebuilt_params=params,
        account_configuration=AccountConfiguration(args.username, args.password),
        server_configuration=server_config,
        battle_format=args.format,
    )

    # Wait for login
    log.info("Waiting for PS login...")
    t_login = time.time()
    while not player.ps_client.logged_in.is_set():
        await asyncio.sleep(0.5)
        if time.time() - t_login > 60:
            log.error("Failed to connect/login after 60s")
            return
    log.info("Logged in! Starting ladder...")

    # ── Play games ──
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
                "search_method": "tree_mcts",
                "simulations": args.simulations,
                "max_depth": args.max_depth,
            }
            elo_history.append(record)

            with open(results_file, "a") as f:
                f.write(json.dumps(record) + "\n")

            elo_str = f"Elo {rating}" if rating else "Elo ?"
            log.info("[%d/%d] %s in %dt | %s | %dW/%dL/%dT (%.0f%%)",
                     total, args.games, result, battle.turn, elo_str,
                     wins, losses, ties, wins / total * 100)

        if total % 10 == 0 and elo_history:
            _plot_elo(elo_history, out_dir, args.simulations)

    # Final summary
    total = wins + losses + ties
    elapsed_min = (time.time() - start) / 60
    log.info("=" * 60)
    log.info("DONE: %dW/%dL/%dT (%.1f%%) in %.0f minutes",
             wins, losses, ties, wins / max(total, 1) * 100, elapsed_min)
    if elo_history and elo_history[-1]["rating"]:
        log.info("Final Elo: %s", elo_history[-1]["rating"])
    log.info("MCTS: %d sims, depth=%d, c_puct=%.2f",
             args.simulations, args.max_depth, args.c_puct)
    log.info("=" * 60)

    with open(out_dir / "ladder_summary.json", "w") as f:
        json.dump({
            "checkpoint": args.checkpoint, "format": args.format,
            "total_games": total, "wins": wins, "losses": losses, "ties": ties,
            "win_rate": round(wins / max(total, 1), 4),
            "final_elo": elo_history[-1]["rating"] if elo_history else None,
            "search_method": "tree_mcts",
            "simulations": args.simulations,
            "c_puct": args.c_puct,
            "opp_temperature": args.opp_temperature,
            "max_depth": args.max_depth,
            "batch_size": args.batch_size,
            "history": elo_history,
        }, f, indent=2)

    _plot_elo(elo_history, out_dir, args.simulations)


def _plot_elo(history, out_dir, n_sims):
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
    ax1.set_title(f"PS Ladder (Tree MCTS {n_sims} sims) - Gen 4 Random Battle\n"
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
