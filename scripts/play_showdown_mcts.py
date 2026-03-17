#!/usr/bin/env python3
"""
Play pokejax model on a local Pokemon Showdown server using GPU-parallel
expectimax search for move selection.

Instead of picking moves from a single neural-network forward pass, this
script converts each game state into the pokejax engine, simulates all
possible (our_action × opponent_action) pairs in parallel on the GPU,
and picks the action with the highest expected value.

Requires a local Pokemon Showdown server running on localhost:8000.
Start it with: cd pokemon-showdown && node pokemon-showdown start --no-security

Usage:
    # Default: 16 samples, vs heuristic
    python scripts/play_showdown_mcts.py --checkpoint checkpoints/ppo_latest.pkl --games 5

    # More samples for better accuracy
    python scripts/play_showdown_mcts.py --checkpoint checkpoints/ppo_latest.pkl \\
        --games 10 --n-samples 64

    # Verbose output
    python scripts/play_showdown_mcts.py --checkpoint checkpoints/ppo_latest.pkl \\
        --games 3 --verbose

    # Compare: search vs policy-only (run play_showdown.py for policy-only)
"""

import argparse
import asyncio
import random
import sys

sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1, closefd=False)


async def main():
    _tag = random.randint(1000, 9999)
    parser = argparse.ArgumentParser(
        description="Play pokejax model on Pokemon Showdown with expectimax search"
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ppo_latest.pkl")
    parser.add_argument("--gen", type=int, default=4)
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--vs", type=str, default="heuristic",
                        choices=["random", "heuristic", "maxpower"])
    parser.add_argument("--format", type=str, default="gen4randombattle")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--server", type=str, default="localhost",
                        help="Showdown server host")
    parser.add_argument("--port", type=int, default=8000,
                        help="Showdown server port")

    # Search parameters
    parser.add_argument("--n-samples", type=int, default=16,
                        help="RNG samples per action pair (default: 16)")
    parser.add_argument("--opp-temperature", type=float, default=0.5,
                        help="Opponent policy temperature (default: 0.5)")

    args = parser.parse_args()

    from poke_env import LocalhostServerConfiguration, AccountConfiguration
    from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer

    try:
        from poke_env.player import MaxBasePowerPlayer
    except ImportError:
        MaxBasePowerPlayer = None

    from pokejax.players.mcts_player import MctsPlayer

    print(f"Loading pokejax model from {args.checkpoint}...")
    print(f"Search config: n_samples={args.n_samples}, "
          f"opp_temp={args.opp_temperature}")

    # Create our player (with search)
    player = MctsPlayer(
        checkpoint_path=args.checkpoint,
        gen=args.gen,
        n_samples=args.n_samples,
        opp_temperature=args.opp_temperature,
        verbose=args.verbose,
        account_configuration=AccountConfiguration(f"MctsBot{_tag}", None),
        server_configuration=LocalhostServerConfiguration,
        battle_format=args.format,
    )

    # Create opponent
    if args.vs == "random":
        opponent = RandomPlayer(
            account_configuration=AccountConfiguration(f"RandomBot{_tag}", None),
            server_configuration=LocalhostServerConfiguration,
            battle_format=args.format,
        )
        opp_name = "RandomPlayer"
    elif args.vs == "maxpower":
        if MaxBasePowerPlayer is None:
            print("MaxBasePowerPlayer not available in this poke-env version")
            return
        opponent = MaxBasePowerPlayer(
            account_configuration=AccountConfiguration(f"MaxPowerBot{_tag}", None),
            server_configuration=LocalhostServerConfiguration,
            battle_format=args.format,
        )
        opp_name = "MaxBasePowerPlayer"
    else:
        opponent = SimpleHeuristicsPlayer(
            account_configuration=AccountConfiguration(f"HeuristicBot{_tag}", None),
            server_configuration=LocalhostServerConfiguration,
            battle_format=args.format,
        )
        opp_name = "SimpleHeuristicsPlayer"

    print(f"\nPlaying {args.games} games: MctsBot vs {opp_name}")
    print(f"Format: {args.format}")
    print(f"Server: {args.server}:{args.port}")
    print()

    # Compile search kernels before playing
    player.warmup()

    # Play games
    await player.battle_against(opponent, n_battles=args.games)

    # Print results
    wins = player.n_won_battles
    losses = player.n_lost_battles
    ties = player.n_tied_battles
    total = wins + losses + ties

    print(f"\n{'=' * 60}")
    print(f"  Search: samples={args.n_samples}, opp_t={args.opp_temperature}")
    print(f"  Results: {wins}W / {losses}L / {ties}T  "
          f"({wins}/{total} = {wins/max(total,1):.0%})")
    print(f"{'=' * 60}")

    # Per-battle results
    for battle_tag, battle in player.battles.items():
        won = battle.won
        turns = battle.turn
        result = "WIN" if won else ("LOSS" if won is False else "TIE")
        print(f"  {battle_tag}: {result} (turn {turns})")


if __name__ == "__main__":
    asyncio.run(main())
