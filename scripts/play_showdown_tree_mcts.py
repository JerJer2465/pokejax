#!/usr/bin/env python3
"""
Play pokejax model on a local Pokemon Showdown server using true MCTS
tree search for move selection.

Requires a local Pokemon Showdown server running on localhost:8000.
Start it with: cd pokemon-showdown && node pokemon-showdown start --no-security

Usage:
    # Default: 128 sims, vs heuristic
    python3 scripts/play_showdown_tree_mcts.py \
        --checkpoint checkpoints/BC-250M/ppo_best.pkl --games 5

    # More sims, verbose
    python3 scripts/play_showdown_tree_mcts.py \
        --checkpoint checkpoints/BC-250M/ppo_best.pkl \
        --games 10 --simulations 256 --verbose

    # Compare search vs no-search:
    #   python3 scripts/play_showdown.py --games 50       (no search)
    #   python3 scripts/play_showdown_tree_mcts.py --games 50  (MCTS)
"""

import argparse
import asyncio
import random
import sys

sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1, closefd=False)


async def main():
    _tag = random.randint(1000, 9999)
    parser = argparse.ArgumentParser(
        description="Play pokejax with MCTS tree search on local PS server"
    )
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/BC-250M/ppo_best.pkl")
    parser.add_argument("--gen", type=int, default=4)
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--vs", type=str, default="heuristic",
                        choices=["random", "heuristic", "maxpower"])
    parser.add_argument("--format", type=str, default="gen4randombattle")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--server", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)

    # MCTS parameters
    parser.add_argument("--simulations", type=int, default=128,
                        help="MCTS simulations per move (default: 128)")
    parser.add_argument("--c-puct", type=float, default=2.5)
    parser.add_argument("--opp-temperature", type=float, default=0.5)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batched", action="store_true",
                        help="Use virtual-loss batched search")

    args = parser.parse_args()

    from poke_env import LocalhostServerConfiguration, AccountConfiguration
    from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer

    try:
        from poke_env.player import MaxBasePowerPlayer
    except ImportError:
        MaxBasePowerPlayer = None

    from pokejax.players.tree_search_player import TreeSearchPlayer

    print(f"Loading pokejax model from {args.checkpoint}...")
    print(f"MCTS config: sims={args.simulations}, c_puct={args.c_puct}, "
          f"depth={args.max_depth}, batch={args.batch_size}")

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

    print(f"\nPlaying {args.games} games: TreeMCTS vs {opp_name}")
    print(f"Format: {args.format}")
    print()

    await player.battle_against(opponent, n_battles=args.games)

    # Print results
    wins = player.n_won_battles
    losses = player.n_lost_battles
    ties = player.n_tied_battles
    total = wins + losses + ties

    print(f"\n{'=' * 60}")
    print(f"  MCTS: sims={args.simulations}, depth={args.max_depth}, "
          f"c_puct={args.c_puct}")
    print(f"  Results: {wins}W / {losses}L / {ties}T  "
          f"({wins}/{total} = {wins/max(total,1):.0%})")
    print(f"{'=' * 60}")

    for battle_tag, battle in player.battles.items():
        won = battle.won
        turns = battle.turn
        result = "WIN" if won else ("LOSS" if won is False else "TIE")
        print(f"  {battle_tag}: {result} (turn {turns})")


if __name__ == "__main__":
    asyncio.run(main())
