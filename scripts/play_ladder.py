#!/usr/bin/env python3
"""
Play pokejax model on the public Pokemon Showdown ladder.

Usage:
    py -3 scripts/play_ladder.py --checkpoint checkpoints/BC-250M/ppo_best.pkl --games 500
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

# Remove proxy env vars so websockets connects directly
# (Clash TUN + DIRECT rules handle routing at the network level)
for k in list(os.environ):
    if k.lower() in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        del os.environ[k]

# Monkey-patch websockets.connect to increase open_timeout.
# Clash TUN's fake-ip DNS can take 10+ seconds to resolve psim.us,
# exceeding the default 10s open_timeout and causing handshake timeouts.
import websockets
import websockets.asyncio.client as _wsc
_orig_connect = _wsc.connect
class _patched_connect(_orig_connect):
    def __init__(self, *a, **kw):
        kw.setdefault("open_timeout", 30)
        super().__init__(*a, **kw)
_wsc.connect = _patched_connect
websockets.connect = _patched_connect

from poke_env import ShowdownServerConfiguration, AccountConfiguration

# Patch poke-env's already-imported ws reference too
import poke_env.ps_client.ps_client as _ps_mod
_ps_mod.ws.connect = _patched_connect

# Monkey-patch poke-env to handle proxy-locked login.
# When PS detects a proxy IP, it returns "‽username" instead of " username"
# in the updateuser message, which prevents poke-env from setting logged_in.
_orig_handle = _ps_mod.PSClient._handle_message

async def _patched_handle(self, message):
    if "|updateuser|" in message:
        message = message.replace("|updateuser|\u203d", "|updateuser| ")
    return await _orig_handle(self, message)

_ps_mod.PSClient._handle_message = _patched_handle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokejax.players.showdown_player import PokejaxPlayer


async def main():
    parser = argparse.ArgumentParser(description="Play pokejax on PS ladder")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/BC-250M/ppo_best.pkl")
    parser.add_argument("--gen", type=int, default=4)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--format", type=str, default="gen4randombattle")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-dir", type=str, default="ladder_results")
    parser.add_argument("--username", type=str, default="paganini2465")
    parser.add_argument("--password", type=str, default="20040605jY")
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
    fh = logging.FileHandler(out_dir / "ladder.log")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    log = logging.getLogger("ladder")

    log.info("Loading model from %s", args.checkpoint)
    log.info("User: %s | Format: %s | Games: %d", args.username, args.format, args.games)

    player = PokejaxPlayer(
        checkpoint_path=args.checkpoint,
        gen=args.gen,
        temperature=args.temperature,
        verbose=args.verbose,
        account_configuration=AccountConfiguration(args.username, args.password),
        server_configuration=ShowdownServerConfiguration,
        battle_format=args.format,
    )

    log.info("Model loaded, starting ladder...")

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
    ax1.set_title(f"PS Ladder - Gen 4 Random Battle\n"
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
