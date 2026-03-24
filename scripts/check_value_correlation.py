#!/usr/bin/env python3
"""
Check whether the model's value function is correctly correlated with game outcomes.

Plays games on a local Pokemon Showdown server, records the value estimate at every
turn, then after each game computes correlation between value trajectories and outcomes.

Usage:
    cd /mnt/c/Users/jerry/Documents/Coding/pokejax
    python3 scripts/check_value_correlation.py --checkpoint checkpoints/BC-250M/ppo_best.pkl --games 20
    python3 scripts/check_value_correlation.py --checkpoint checkpoints/BC-250M/ppo_best.pkl --games 50 --vs heuristic
"""

import argparse
import asyncio
import random
import sys
import os
from collections import defaultdict

import numpy as np

sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1, closefd=False)


# ---------------------------------------------------------------------------
# Instrumented player that records per-turn values
# ---------------------------------------------------------------------------

class ValueTrackingPlayer:
    """Mixin that patches choose_move to record value estimates per turn."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # battle_tag -> list of (turn, value) tuples
        self._value_history: dict = defaultdict(list)

    def choose_move(self, battle):
        # Run the parent's obs building + forward pass to get the value
        obs = self.obs_bridge.build_obs(battle)

        import jax.numpy as jnp
        int_ids   = jnp.array(obs["int_ids"])
        float_feats = jnp.array(obs["float_feats"])
        legal_mask  = jnp.array(obs["legal_mask"])

        _, value = self._forward(self.params, int_ids, float_feats, legal_mask)
        self._value_history[battle.battle_tag].append((battle.turn, float(value)))

        # Delegate actual move selection to the real player
        return super().choose_move(battle)


async def main():
    _tag = random.randint(1000, 9999)

    parser = argparse.ArgumentParser(description="Check value function correlation on local PS server")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/BC-250M/ppo_best.pkl")
    parser.add_argument("--games",      type=int, default=20)
    parser.add_argument("--vs",         type=str, default="heuristic",
                        choices=["random", "heuristic", "maxpower"])
    parser.add_argument("--format",     type=str, default="gen4randombattle")
    parser.add_argument("--gen",        type=int, default=4)
    parser.add_argument("--plot",       action="store_true",
                        help="Save value trajectory plots to value_plots/")
    args = parser.parse_args()

    from poke_env import LocalhostServerConfiguration, AccountConfiguration
    from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer

    from pokejax.players.showdown_player import PokejaxPlayer

    # Build an instrumented subclass dynamically
    class TrackedPokejaxPlayer(ValueTrackingPlayer, PokejaxPlayer):
        pass

    print(f"Loading checkpoint: {args.checkpoint}")
    player = TrackedPokejaxPlayer(
        checkpoint_path=args.checkpoint,
        gen=args.gen,
        temperature=0.0,
        verbose=False,
        account_configuration=AccountConfiguration(f"ValBot{_tag}", None),
        server_configuration=LocalhostServerConfiguration,
        battle_format=args.format,
    )

    if args.vs == "random":
        opponent = RandomPlayer(
            account_configuration=AccountConfiguration(f"RandOpp{_tag}", None),
            server_configuration=LocalhostServerConfiguration,
            battle_format=args.format,
        )
        opp_name = "Random"
    elif args.vs == "maxpower":
        from poke_env.player import MaxBasePowerPlayer
        opponent = MaxBasePowerPlayer(
            account_configuration=AccountConfiguration(f"MaxOpp{_tag}", None),
            server_configuration=LocalhostServerConfiguration,
            battle_format=args.format,
        )
        opp_name = "MaxBasePower"
    else:
        opponent = SimpleHeuristicsPlayer(
            account_configuration=AccountConfiguration(f"HeurOpp{_tag}", None),
            server_configuration=LocalhostServerConfiguration,
            battle_format=args.format,
        )
        opp_name = "Heuristic"

    print(f"Playing {args.games} games vs {opp_name} ({args.format})")
    print()

    await player.battle_against(opponent, n_battles=args.games)

    # ---------------------------------------------------------------------------
    # Analysis
    # ---------------------------------------------------------------------------
    wins   = player.n_won_battles
    losses = player.n_lost_battles
    ties   = player.n_tied_battles
    total  = wins + losses + ties

    print(f"{'='*60}")
    print(f"  Outcome: {wins}W / {losses}L / {ties}T  (win rate {wins/max(total,1):.0%})")
    print(f"{'='*60}\n")

    # Per-game outcome label: +1 = win, -1 = loss, 0 = tie
    outcomes = {}
    for btag, battle in player.battles.items():
        if battle.won is True:
            outcomes[btag] = 1.0
        elif battle.won is False:
            outcomes[btag] = -1.0
        else:
            outcomes[btag] = 0.0

    # ---------------------------------------------------------------------------
    # Metric 1: Correlation between FINAL value and outcome
    # ---------------------------------------------------------------------------
    final_values = []
    outcome_labels = []
    for btag, history in player._value_history.items():
        if not history:
            continue
        final_val = history[-1][1]
        out = outcomes.get(btag, 0.0)
        final_values.append(final_val)
        outcome_labels.append(out)

    final_values    = np.array(final_values)
    outcome_labels  = np.array(outcome_labels)

    if len(final_values) > 1:
        corr = np.corrcoef(final_values, outcome_labels)[0, 1]
        print(f"[Metric 1] Final-value vs outcome Pearson r = {corr:.4f}")
        print(f"  (r close to +1 means model correctly assigns high value to wins)")
    else:
        corr = float('nan')
        print("[Metric 1] Not enough games for correlation")

    # ---------------------------------------------------------------------------
    # Metric 2: Mean initial / middle / final value split by outcome
    # ---------------------------------------------------------------------------
    print()
    print("[Metric 2] Mean value at different game phases (W vs L)")
    win_init, win_mid, win_final = [], [], []
    los_init, los_mid, los_final = [], [], []

    for btag, history in player._value_history.items():
        if len(history) < 2:
            continue
        vals = [v for _, v in history]
        n = len(vals)
        v_init  = vals[0]
        v_mid   = vals[n // 2]
        v_final = vals[-1]
        out = outcomes.get(btag, 0.0)
        if out > 0:
            win_init.append(v_init);  win_mid.append(v_mid);  win_final.append(v_final)
        elif out < 0:
            los_init.append(v_init);  los_mid.append(v_mid);  los_final.append(v_final)

    def _fmt(lst):
        if not lst:
            return "  n/a"
        return f"{np.mean(lst):+.3f} ± {np.std(lst):.3f} (n={len(lst)})"

    print(f"  {'Phase':<10}  {'WIN':>25}  {'LOSS':>25}")
    print(f"  {'Initial':<10}  {_fmt(win_init):>25}  {_fmt(los_init):>25}")
    print(f"  {'Middle':<10}  {_fmt(win_mid):>25}  {_fmt(los_mid):>25}")
    print(f"  {'Final':<10}  {_fmt(win_final):>25}  {_fmt(los_final):>25}")

    # ---------------------------------------------------------------------------
    # Metric 3: Value monotonicity — does value trend in the right direction?
    # ---------------------------------------------------------------------------
    print()
    print("[Metric 3] Value trajectory monotonicity")
    consistent_wins  = 0
    consistent_loses = 0
    total_wins_check = 0
    total_loss_check = 0

    for btag, history in player._value_history.items():
        if len(history) < 4:
            continue
        vals = [v for _, v in history]
        # Check whether the last-quarter average > first-quarter average
        n = len(vals)
        q = max(1, n // 4)
        early_mean = np.mean(vals[:q])
        late_mean  = np.mean(vals[-q:])
        out = outcomes.get(btag, 0.0)
        if out > 0:
            total_wins_check += 1
            if late_mean > early_mean:  # value should go up in winning games
                consistent_wins += 1
        elif out < 0:
            total_loss_check += 1
            if late_mean < early_mean:  # value should go down in losing games
                consistent_loses += 1

    if total_wins_check:
        print(f"  Wins where value rose   late > early: "
              f"{consistent_wins}/{total_wins_check} "
              f"({consistent_wins/total_wins_check:.0%})")
    if total_loss_check:
        print(f"  Losses where value fell late < early: "
              f"{consistent_loses}/{total_loss_check} "
              f"({consistent_loses/total_loss_check:.0%})")

    # ---------------------------------------------------------------------------
    # Metric 4: Per-game summary table
    # ---------------------------------------------------------------------------
    print()
    print("[Metric 4] Per-game summary")
    print(f"  {'Game':<20}  {'Result':<6}  {'Init':>6}  {'Mid':>6}  {'Final':>6}  {'Turns':>5}")
    print(f"  {'-'*20}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*5}")
    for btag, history in sorted(player._value_history.items()):
        if not history:
            continue
        vals   = [v for _, v in history]
        turns  = history[-1][0]
        out    = outcomes.get(btag, 0.0)
        result = "WIN" if out > 0 else ("LOSS" if out < 0 else "TIE ")
        n = len(vals)
        v_init  = vals[0]
        v_mid   = vals[n // 2]
        v_final = vals[-1]
        print(f"  {btag[-20:]:<20}  {result:<6}  {v_init:+.3f}  {v_mid:+.3f}  {v_final:+.3f}  {turns:>5}")

    # ---------------------------------------------------------------------------
    # Optional: plot value trajectories
    # ---------------------------------------------------------------------------
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            os.makedirs("value_plots", exist_ok=True)

            # Aggregate plot
            fig, ax = plt.subplots(figsize=(10, 5))
            for btag, history in player._value_history.items():
                if len(history) < 2:
                    continue
                turns = [t for t, _ in history]
                vals  = [v for _, v in history]
                out   = outcomes.get(btag, 0.0)
                color = "green" if out > 0 else ("red" if out < 0 else "gray")
                ax.plot(turns, vals, color=color, alpha=0.5, linewidth=1)

            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
            ax.set_xlabel("Turn")
            ax.set_ylabel("Value estimate")
            ax.set_title(f"Value trajectories (green=win, red=loss) | r={corr:.3f}")
            # Custom legend
            from matplotlib.lines import Line2D
            ax.legend(handles=[
                Line2D([0], [0], color='green', label=f'Win ({wins})'),
                Line2D([0], [0], color='red',   label=f'Loss ({losses})'),
                Line2D([0], [0], color='gray',  label=f'Tie ({ties})'),
            ])
            fig.tight_layout()
            fig.savefig("value_plots/trajectories.png", dpi=120)
            print(f"\nPlot saved to value_plots/trajectories.png")
            plt.close(fig)
        except ImportError:
            print("\n(matplotlib not available, skipping plots)")

    # ---------------------------------------------------------------------------
    # Summary verdict
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  VERDICT")
    if not np.isnan(corr):
        if corr > 0.5:
            verdict = "GOOD — value strongly correlated with outcomes"
        elif corr > 0.2:
            verdict = "WEAK — some correlation, but value is noisy"
        elif corr > -0.1:
            verdict = "POOR — near-zero correlation, value may be miscalibrated"
        else:
            verdict = "BAD  — negative correlation, value is inverted!"
        print(f"  Final-value correlation r={corr:.3f}: {verdict}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
