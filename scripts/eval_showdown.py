#!/usr/bin/env python3
"""
Evaluate a PPO checkpoint on a local Pokemon Showdown server.

Run periodically during training to benchmark against real Showdown opponents.

Prerequisites:
    1. Start local Showdown server:
       cd pokemon-showdown && node pokemon-showdown start --no-security

    2. Run from Windows pokeEnv conda env:
       conda activate pokeEnv
       python scripts/eval_showdown.py --checkpoint checkpoints/ppo_latest.pkl --games 20

Usage:
    # Quick eval (5 games vs each opponent):
    python scripts/eval_showdown.py --checkpoint checkpoints/ppo_latest.pkl

    # Full eval (20 games vs each):
    python scripts/eval_showdown.py --checkpoint checkpoints/ppo_latest.pkl --games 20

    # Watch games:
    python scripts/eval_showdown.py --checkpoint checkpoints/ppo_latest.pkl --verbose

    # Monitor latest checkpoint continuously:
    python scripts/eval_showdown.py --checkpoint checkpoints/ppo_latest.pkl --watch --interval 300
"""

import argparse
import asyncio
import os
import sys
import time

sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1, closefd=False)


async def run_eval(checkpoint_path, n_games, vs_type, gen, battle_format,
                   server, port, verbose, temperature):
    """Run eval games and return results dict."""
    from poke_env import LocalhostServerConfiguration, AccountConfiguration
    from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
    from pokejax.players.showdown_player import PokejaxPlayer

    player = PokejaxPlayer(
        checkpoint_path=checkpoint_path,
        gen=gen,
        temperature=temperature,
        verbose=verbose,
        account_configuration=AccountConfiguration("PokejaxEval", None),
        server_configuration=LocalhostServerConfiguration,
        battle_format=battle_format,
    )

    if vs_type == "random":
        opponent = RandomPlayer(
            account_configuration=AccountConfiguration("EvalRandom", None),
            server_configuration=LocalhostServerConfiguration,
            battle_format=battle_format,
        )
        opp_name = "Random"
    else:
        opponent = SimpleHeuristicsPlayer(
            account_configuration=AccountConfiguration("EvalHeuristic", None),
            server_configuration=LocalhostServerConfiguration,
            battle_format=battle_format,
        )
        opp_name = "Heuristic"

    await player.battle_against(opponent, n_battles=n_games)

    wins = player.n_won_battles
    losses = player.n_lost_battles
    total = wins + losses + player.n_tied_battles
    win_rate = wins / max(total, 1)

    return {
        "opponent": opp_name,
        "wins": wins,
        "losses": losses,
        "total": total,
        "win_rate": win_rate,
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PPO checkpoint on Pokemon Showdown",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ppo_latest.pkl")
    parser.add_argument("--gen", type=int, default=4)
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--format", type=str, default="gen4randombattle")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--server", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--watch", action="store_true",
                        help="Continuously watch and eval latest checkpoint")
    parser.add_argument("--interval", type=int, default=300,
                        help="Seconds between watch evals")
    args = parser.parse_args()

    if args.watch:
        print(f"Watching {args.checkpoint} for changes (eval every {args.interval}s)")
        last_mtime = 0
        while True:
            if os.path.exists(args.checkpoint):
                mtime = os.path.getmtime(args.checkpoint)
                if mtime > last_mtime:
                    last_mtime = mtime
                    print(f"\n{'='*60}")
                    print(f"New checkpoint detected: {time.ctime(mtime)}")
                    print(f"{'='*60}")

                    for vs in ["heuristic", "random"]:
                        result = await run_eval(
                            args.checkpoint, args.games, vs, args.gen,
                            args.format, args.server, args.port,
                            args.verbose, args.temperature,
                        )
                        print(f"  vs {result['opponent']}: "
                              f"{result['wins']}W/{result['losses']}L "
                              f"({result['win_rate']:.0%})")
            else:
                print(f"Waiting for {args.checkpoint}...")

            await asyncio.sleep(args.interval)
    else:
        if not os.path.exists(args.checkpoint):
            print(f"Checkpoint not found: {args.checkpoint}")
            return

        print(f"Evaluating: {args.checkpoint}")
        print(f"Format: {args.format}, Games: {args.games}")
        print()

        for vs in ["heuristic", "random"]:
            result = await run_eval(
                args.checkpoint, args.games, vs, args.gen,
                args.format, args.server, args.port,
                args.verbose, args.temperature,
            )
            print(f"vs {result['opponent']}: "
                  f"{result['wins']}W/{result['losses']}L "
                  f"({result['win_rate']:.0%})")


if __name__ == "__main__":
    asyncio.run(main())
