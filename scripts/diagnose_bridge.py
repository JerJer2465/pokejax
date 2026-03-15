#!/usr/bin/env python3
"""
Diagnostic script: compare bridge observations against expected engine behavior.
Plays a few games on Showdown and logs detailed per-turn diagnostics.
"""
import argparse
import asyncio
import sys
import traceback

sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1, closefd=False)

import numpy as np


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/bc_final.pkl")
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--gen", type=int, default=4)
    args = parser.parse_args()

    from poke_env import LocalhostServerConfiguration, AccountConfiguration
    from poke_env.player import RandomPlayer
    from poke_env.environment import AbstractBattle, Pokemon, Move

    from pokejax.players.showdown_player import PokejaxPlayer, ObsBridge
    import jax.numpy as jnp

    # Patch PokejaxPlayer to add diagnostics
    original_choose_move_impl = PokejaxPlayer._choose_move_impl

    turn_log = []

    def diagnostic_choose_move(self, battle: AbstractBattle):
        available_moves = battle.available_moves
        available_switches = battle.available_switches
        own_active = battle.active_pokemon
        opp_active = battle.opponent_active_pokemon

        info = {
            "turn": battle.turn,
            "n_available_moves": len(available_moves),
            "n_available_switches": len(available_switches),
            "trapped": getattr(battle, 'trapped', False),
            "force_switch": getattr(battle, 'force_switch', False),
        }

        # Check active pokemon
        if own_active:
            info["own_active_species"] = own_active.species
            info["own_active_hp"] = f"{own_active.current_hp}/{own_active.max_hp}"
            info["own_active_fainted"] = own_active.fainted
            info["own_active_n_moves"] = len(own_active.moves) if own_active.moves else 0
        else:
            info["own_active_species"] = None

        if opp_active:
            info["opp_active_species"] = opp_active.species
            info["opp_active_hp_frac"] = opp_active.current_hp_fraction
        else:
            info["opp_active_species"] = None

        # Check move identity matching
        if own_active and own_active.moves:
            own_move_list = list(own_active.moves.values())[:4]
            available_move_ids = set(id(m) for m in available_moves)
            identity_matches = []
            for i, m in enumerate(own_move_list):
                matched = id(m) in available_move_ids
                identity_matches.append(matched)
                if not matched and m is not None:
                    # Check if there's a matching move by ID
                    name_match = any(am.id == m.id for am in available_moves)
                    if name_match:
                        info[f"move_{i}_identity_FAIL_but_name_match"] = m.id
            info["move_identity_matches"] = identity_matches
            info["move_names"] = [m.id if m else None for m in own_move_list]
            info["available_move_names"] = [m.id for m in available_moves]

        # Check switch identity matching
        own_team = list(battle.team.values())
        while len(own_team) < 6:
            own_team.append(None)
        own_team = own_team[:6]
        available_switch_ids = set(id(p) for p in available_switches)
        switch_identity = []
        for slot in range(6):
            p = own_team[slot]
            if p is not None:
                matched = id(p) in available_switch_ids
                switch_identity.append((p.species, matched, p.fainted, p == own_active))
            else:
                switch_identity.append((None, False, False, False))
        info["switch_identity"] = switch_identity

        # Build obs and check legal mask
        obs = self.obs_bridge.build_obs(battle)
        info["legal_mask"] = obs["legal_mask"].tolist()
        info["legal_mask_sum"] = float(obs["legal_mask"].sum())

        # Check int_ids for own team tokens (1-6)
        own_species_ids = []
        for tok in range(1, 7):
            sid = int(obs["int_ids"][tok, 0])
            own_species_ids.append(sid)
        info["own_species_ids"] = own_species_ids

        # Check int_ids for opp team tokens (7-12)
        opp_species_ids = []
        for tok in range(7, 13):
            sid = int(obs["int_ids"][tok, 0])
            opp_species_ids.append(sid)
        info["opp_species_ids"] = opp_species_ids

        # Check active own mon's move int IDs (token with is_active=1)
        for tok in range(1, 7):
            if obs["float_feats"][tok, 179] > 0.5:  # _OFF_IS_ACTIVE
                info["active_token"] = tok
                info["active_move_ids"] = [int(obs["int_ids"][tok, m+1]) for m in range(4)]
                info["active_hp_frac"] = float(obs["float_feats"][tok, 0])
                info["active_hp_bin"] = obs["float_feats"][tok, 1:11].tolist()
                info["active_base_stats"] = obs["float_feats"][tok, 11:17].tolist()
                info["active_type1"] = obs["float_feats"][tok, 142:160].tolist()
                info["active_slot_onehot"] = obs["float_feats"][tok, 180:186].tolist()
                break

        # Run model
        int_ids = jnp.array(obs["int_ids"])
        float_feats = jnp.array(obs["float_feats"])
        legal_mask = jnp.array(obs["legal_mask"])
        log_probs, value = self._forward(self.params, int_ids, float_feats, legal_mask)
        probs = np.exp(np.array(log_probs))

        info["value"] = float(value)
        info["action_probs"] = probs.tolist()
        info["top_action"] = int(np.argmax(probs * np.array(obs["legal_mask"])))

        turn_log.append(info)

        # Now call original
        return original_choose_move_impl(self, battle)

    PokejaxPlayer._choose_move_impl = diagnostic_choose_move

    # Create players
    player = PokejaxPlayer(
        checkpoint_path=args.checkpoint,
        gen=args.gen,
        temperature=0.0,
        verbose=False,
        account_configuration=AccountConfiguration("DiagBot", None),
        server_configuration=LocalhostServerConfiguration,
        battle_format="gen4randombattle",
    )
    opponent = RandomPlayer(
        account_configuration=AccountConfiguration("RandBot", None),
        server_configuration=LocalhostServerConfiguration,
        battle_format="gen4randombattle",
    )

    print(f"Playing {args.games} diagnostic games...")
    await player.battle_against(opponent, n_battles=args.games)

    wins = player.n_won_battles
    losses = player.n_lost_battles
    total = wins + losses
    print(f"\nResults: {wins}W / {losses}L ({wins}/{total} = {wins/max(total,1):.0%})")

    # Analyze diagnostics
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC ANALYSIS ({len(turn_log)} choose_move calls)")
    print(f"{'='*70}")

    # 1. Legal mask issues
    n_zero_mask = sum(1 for t in turn_log if t["legal_mask_sum"] <= 1.0)
    n_no_moves_legal = sum(1 for t in turn_log if sum(t["legal_mask"][:4]) == 0)
    n_no_switches_legal = sum(1 for t in turn_log if sum(t["legal_mask"][4:]) == 0)
    print(f"\nLegal mask:")
    print(f"  Only 1 action legal: {n_zero_mask}/{len(turn_log)}")
    print(f"  No moves legal (moves all 0): {n_no_moves_legal}/{len(turn_log)}")
    print(f"  No switches legal: {n_no_switches_legal}/{len(turn_log)}")

    # 2. Identity matching failures
    n_identity_fail = 0
    n_identity_total = 0
    for t in turn_log:
        if "move_identity_matches" in t:
            for i, matched in enumerate(t["move_identity_matches"]):
                n_identity_total += 1
                if not matched:
                    n_identity_fail += 1
    print(f"\nMove identity matching:")
    print(f"  Failures: {n_identity_fail}/{n_identity_total}")

    # Show all identity failures
    for t in turn_log:
        for key in t:
            if "identity_FAIL" in key:
                print(f"    Turn {t['turn']}: {key} = {t[key]}")

    # 3. Force switches
    n_force = sum(1 for t in turn_log if t.get("force_switch"))
    print(f"\nForce switches: {n_force}/{len(turn_log)}")

    # 4. No active pokemon
    n_no_active = sum(1 for t in turn_log if t.get("own_active_species") is None)
    print(f"No active pokemon: {n_no_active}/{len(turn_log)}")

    # 5. Action distribution
    action_counts = [0] * 10
    for t in turn_log:
        action_counts[t["top_action"]] += 1
    print(f"\nAction distribution:")
    for a in range(10):
        if action_counts[a] > 0:
            pct = action_counts[a] / len(turn_log) * 100
            label = f"move_{a}" if a < 4 else f"switch_{a-4}"
            print(f"  {label}: {action_counts[a]} ({pct:.1f}%)")

    # 6. Value estimates
    values = [t["value"] for t in turn_log]
    print(f"\nModel value estimates:")
    print(f"  Mean: {np.mean(values):.3f}")
    print(f"  Min: {np.min(values):.3f}")
    print(f"  Max: {np.max(values):.3f}")

    # 7. Species ID verification
    n_zero_species = 0
    for t in turn_log:
        if t.get("own_active_species"):
            if t["own_species_ids"] and all(s == 0 for s in t["own_species_ids"]):
                n_zero_species += 1
    print(f"\nAll own species IDs = 0: {n_zero_species}/{len(turn_log)}")

    # 8. Print first few turns in detail
    print(f"\n{'='*70}")
    print("DETAILED TURN LOG (first 10 turns)")
    print(f"{'='*70}")
    for t in turn_log[:10]:
        print(f"\n--- Turn {t['turn']} ---")
        print(f"  Own active: {t.get('own_active_species', 'None')} "
              f"HP={t.get('own_active_hp', '?')}")
        print(f"  Opp active: {t.get('opp_active_species', 'None')} "
              f"HP_frac={t.get('opp_active_hp_frac', '?')}")
        print(f"  Available moves ({t['n_available_moves']}): "
              f"{t.get('available_move_names', [])}")
        print(f"  Available switches: {t['n_available_switches']}")
        print(f"  Legal mask: {t['legal_mask']}")
        if "move_identity_matches" in t:
            print(f"  Move names: {t.get('move_names', [])}")
            print(f"  Identity matches: {t['move_identity_matches']}")
        if "active_move_ids" in t:
            print(f"  Active move IDs: {t['active_move_ids']}")
            print(f"  Active HP frac: {t.get('active_hp_frac', '?'):.3f}")
            print(f"  Active slot: {t.get('active_slot_onehot', [])}")
        print(f"  Value: {t['value']:.3f}")
        print(f"  Top action: {t['top_action']} "
              f"(prob={t['action_probs'][t['top_action']]*100:.1f}%)")
        # Show top 3 actions
        legal = [i for i in range(10) if t['legal_mask'][i] > 0]
        sorted_actions = sorted(legal, key=lambda a: -t['action_probs'][a])
        for a in sorted_actions[:5]:
            label = f"move_{a}" if a < 4 else f"switch_{a-4}"
            print(f"    {label}: {t['action_probs'][a]*100:.1f}%")

    # 9. Print all turns where model chose a switch
    print(f"\n{'='*70}")
    print("SWITCH DECISIONS")
    print(f"{'='*70}")
    n_switch = 0
    for t in turn_log:
        if t["top_action"] >= 4:
            n_switch += 1
            if n_switch <= 15:
                print(f"  Turn {t['turn']}: "
                      f"switch_{t['top_action']-4} "
                      f"(own={t.get('own_active_species','?')}, "
                      f"opp={t.get('opp_active_species','?')}, "
                      f"prob={t['action_probs'][t['top_action']]*100:.1f}%)")
    print(f"  Total switches: {n_switch}/{len(turn_log)} ({n_switch/max(len(turn_log),1)*100:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
