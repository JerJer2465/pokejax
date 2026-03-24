#!/usr/bin/env python3
"""
Play pokejax model with MCTS tree search vs poke-env heuristics on a local PS server.

Combines TreeSearchPlayer (MCTS) with detailed per-game logging, JSONL output,
and win rate graphs.

Prerequisites:
    Start local Showdown server:
        cd pokemon-showdown && node pokemon-showdown start --no-security

Usage:
    python3 scripts/play_local_heuristic_mcts.py --checkpoint checkpoints/ppo_latest.pkl --games 100 --verbose
    python3 scripts/play_local_heuristic_mcts.py --checkpoint checkpoints/BC-250M/ppo_best.pkl --games 50 --simulations 256
"""

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from pathlib import Path

sys.stdout = open(sys.stdout.fileno(), "w", buffering=1, closefd=False)


async def run_games(args, log, out_dir):
    """Play games and return summary dict."""
    from poke_env import LocalhostServerConfiguration, AccountConfiguration
    from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer

    from pokejax.players.tree_search_player import TreeSearchPlayer

    _tag = random.randint(1000, 9999)

    log.info("Loading model from %s", args.checkpoint)
    log.info(
        "MCTS config: sims=%d, c_puct=%.1f, depth=%d, opp_temp=%.2f, batch=%d",
        args.simulations, args.c_puct, args.max_depth,
        args.opp_temperature, args.batch_size,
    )

    player = TreeSearchPlayer(
        checkpoint_path=args.checkpoint,
        gen=args.gen,
        n_simulations=args.simulations,
        c_puct=args.c_puct,
        opp_temperature=args.opp_temperature,
        max_depth=args.max_depth,
        batch_size=args.batch_size,
        use_batched=args.batched,
        verbose=args.verbose,
        account_configuration=AccountConfiguration(f"TreeMCTS{_tag}", None),
        server_configuration=LocalhostServerConfiguration,
        battle_format=args.format,
    )

    if args.vs == "random":
        opponent = RandomPlayer(
            account_configuration=AccountConfiguration(f"RandomBot{_tag}", None),
            server_configuration=LocalhostServerConfiguration,
            battle_format=args.format,
        )
        opp_name = "RandomPlayer"
    else:
        opponent = SimpleHeuristicsPlayer(
            account_configuration=AccountConfiguration(f"HeuristicBot{_tag}", None),
            server_configuration=LocalhostServerConfiguration,
            battle_format=args.format,
        )
        opp_name = "SimpleHeuristicsPlayer"

    log.info("Playing %d games: TreeMCTS vs %s", args.games, opp_name)
    log.info("Format: %s | Server: localhost:8000", args.format)

    results_file = out_dir / "local_mcts_results.jsonl"
    history = []
    wins, losses, ties = 0, 0, 0
    seen_battles = set()
    start = time.time()
    total = 0

    while total < args.games:
        batch = min(args.game_batch, args.games - total)
        try:
            await player.battle_against(opponent, n_battles=batch)
        except Exception as e:
            log.error("Battle error: %s", e, exc_info=True)
            await asyncio.sleep(2)
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
            record = {
                "game": total,
                "battle_tag": tag,
                "result": result,
                "turns": battle.turn,
                "opponent": opp_name,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "win_rate": round(wins / max(total, 1), 4),
                "elapsed_s": round(time.time() - start, 1),
                "mcts_sims": args.simulations,
            }
            history.append(record)

            with open(results_file, "a") as f:
                f.write(json.dumps(record) + "\n")

            log.info(
                "[%d/%d] %s in %dt | %dW/%dL/%dT (%.0f%%)",
                total, args.games, result, battle.turn,
                wins, losses, ties, wins / total * 100,
            )

        if total % 10 == 0 and history:
            _plot_winrate(history, opp_name, args, out_dir)

    elapsed_min = (time.time() - start) / 60
    log.info("=" * 60)
    log.info(
        "DONE: %dW/%dL/%dT (%.1f%%) in %.1f minutes vs %s",
        wins, losses, ties, wins / max(total, 1) * 100, elapsed_min, opp_name,
    )
    log.info(
        "MCTS: sims=%d, c_puct=%.1f, depth=%d",
        args.simulations, args.c_puct, args.max_depth,
    )
    log.info("=" * 60)

    summary = {
        "checkpoint": args.checkpoint,
        "format": args.format,
        "opponent": opp_name,
        "mcts_simulations": args.simulations,
        "mcts_c_puct": args.c_puct,
        "mcts_max_depth": args.max_depth,
        "mcts_opp_temperature": args.opp_temperature,
        "total_games": total,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": round(wins / max(total, 1), 4),
        "elapsed_minutes": round(elapsed_min, 1),
        "history": history,
    }

    with open(out_dir / "local_mcts_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _plot_winrate(history, opp_name, args, out_dir)

    return summary


def _plot_winrate(history, opp_name, args, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not history:
        return

    games = [r["game"] for r in history]
    win_rates = [r["win_rate"] * 100 for r in history]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]}
    )

    ax1.plot(games, win_rates, "b-", linewidth=1.5, alpha=0.8)
    ax1.scatter(games, win_rates, c="blue", s=8, alpha=0.4, zorder=3)
    ax1.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="50%")
    ax1.set_xlabel("Game Number")
    ax1.set_ylabel("Cumulative Win Rate (%)")
    ax1.set_title(
        f"Local MCTS Eval vs {opp_name} (sims={args.simulations})\n"
        f"Win Rate: {win_rates[-1]:.1f}% after {games[-1]} games"
    )
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    window = 10
    rolling = []
    w = 0
    results = [r["result"] for r in history]
    for i, r in enumerate(results):
        if r == "WIN":
            w += 1
        if i >= window:
            if results[i - window] == "WIN":
                w -= 1
            rolling.append(w / window * 100)
        elif i + 1 >= window:
            rolling.append(w / window * 100)

    if rolling:
        ax2.plot(games[window - 1:], rolling, "g-", linewidth=1.5)
        ax2.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Game Number")
        ax2.set_ylabel(f"Rolling Win Rate (last {window} games) %")
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "local_mcts_winrate.png", dpi=150)
    plt.close()


async def main():
    parser = argparse.ArgumentParser(
        description="Play pokejax with MCTS tree search vs heuristics on local PS server"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/BC-250M/ppo_best.pkl"
    )
    parser.add_argument("--gen", type=int, default=4)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument(
        "--vs", type=str, default="heuristic", choices=["random", "heuristic"]
    )
    parser.add_argument("--format", type=str, default="gen4randombattle")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-dir", type=str, default="local_eval_results")
    parser.add_argument(
        "--game-batch", type=int, default=1,
        help="Games per battle_against call (1 = per-game logging)"
    )

    # MCTS parameters
    parser.add_argument("--simulations", type=int, default=128,
                        help="MCTS simulations per move (default: 128)")
    parser.add_argument("--c-puct", type=float, default=2.5)
    parser.add_argument("--opp-temperature", type=float, default=0.5)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Virtual-loss batch size for MCTS")
    parser.add_argument("--batched", action="store_true",
                        help="Use virtual-loss batched search")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    fh = logging.FileHandler(out_dir / "local_mcts_eval.log")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    log = logging.getLogger("local_mcts_eval")

    summary = await run_games(args, log, out_dir)
    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
