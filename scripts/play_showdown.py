#!/usr/bin/env python3
"""
Play pokejax BC/PPO model on a local Pokemon Showdown server.

Requires a local Pokemon Showdown server running on localhost:8000.
Start it with: cd pokemon-showdown && node pokemon-showdown start --no-security

Usage:
    # vs built-in heuristic (SimpleHeuristicsPlayer):
    python scripts/play_showdown.py --checkpoint checkpoints/bc_final.pkl --games 5

    # vs random:
    python scripts/play_showdown.py --checkpoint checkpoints/bc_final.pkl --vs random --games 10

    # vs MaxBasePower:
    python scripts/play_showdown.py --checkpoint checkpoints/bc_final.pkl --vs maxpower --games 5

    # verbose (show action probabilities):
    python scripts/play_showdown.py --checkpoint checkpoints/bc_final.pkl --games 3 --verbose
"""

import argparse
import asyncio
import sys

sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1, closefd=False)


async def main():
    parser = argparse.ArgumentParser(description="Play pokejax model on Pokemon Showdown")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/bc_final.pkl")
    parser.add_argument("--gen", type=int, default=4)
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--vs", type=str, default="heuristic",
                        choices=["random", "heuristic", "maxpower"])
    parser.add_argument("--format", type=str, default="gen4randombattle")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--server", type=str, default="localhost",
                        help="Showdown server host")
    parser.add_argument("--port", type=int, default=8000,
                        help="Showdown server port")
    args = parser.parse_args()

    from poke_env import LocalhostServerConfiguration, AccountConfiguration
    from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer

    # Import MaxBasePowerPlayer if available
    try:
        from poke_env.player import MaxBasePowerPlayer
    except ImportError:
        MaxBasePowerPlayer = None

    from pokejax.players.showdown_player import PokejaxPlayer

    print(f"Loading pokejax model from {args.checkpoint}...")

    # Create our player
    player = PokejaxPlayer(
        checkpoint_path=args.checkpoint,
        gen=args.gen,
        temperature=args.temperature,
        verbose=args.verbose,
        account_configuration=AccountConfiguration("PokejaxBot", None),
        server_configuration=LocalhostServerConfiguration,
        battle_format=args.format,
    )

    # Create opponent
    if args.vs == "random":
        opponent = RandomPlayer(
            account_configuration=AccountConfiguration("RandomBot", None),
            server_configuration=LocalhostServerConfiguration,
            battle_format=args.format,
        )
        opp_name = "RandomPlayer"
    elif args.vs == "maxpower":
        if MaxBasePowerPlayer is None:
            print("MaxBasePowerPlayer not available in this poke-env version")
            return
        opponent = MaxBasePowerPlayer(
            account_configuration=AccountConfiguration("MaxPowerBot", None),
            server_configuration=LocalhostServerConfiguration,
            battle_format=args.format,
        )
        opp_name = "MaxBasePowerPlayer"
    else:
        opponent = SimpleHeuristicsPlayer(
            account_configuration=AccountConfiguration("HeuristicBot", None),
            server_configuration=LocalhostServerConfiguration,
            battle_format=args.format,
        )
        opp_name = "SimpleHeuristicsPlayer"

    print(f"Playing {args.games} games: PokejaxBot vs {opp_name}")
    print(f"Format: {args.format}")
    print(f"Server: {args.server}:{args.port}")
    print()

    # Play games
    await player.battle_against(opponent, n_battles=args.games)

    # Print results
    wins = player.n_won_battles
    losses = player.n_lost_battles
    ties = player.n_tied_battles
    total = wins + losses + ties

    print(f"\n{'=' * 50}")
    print(f"  Results: {wins}W / {losses}L / {ties}T  ({wins}/{total} = {wins/max(total,1):.0%})")
    print(f"{'=' * 50}")

    # Print per-battle results
    for battle_tag, battle in player.battles.items():
        won = battle.won
        turns = battle.turn
        result = "WIN" if won else ("LOSS" if won is False else "TIE")
        print(f"  {battle_tag}: {result} (turn {turns})")


if __name__ == "__main__":
    asyncio.run(main())
