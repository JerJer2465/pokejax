#!/usr/bin/env python3
"""
Diagnostic script: monitor model inputs/outputs during poke-env games.

Logs per-turn:
  - Active pokemon and available moves/switches
  - Legal mask (bridge vs poke-env)
  - Model value estimate and action probabilities
  - Chosen action and what it maps to
  - Float feature summary (non-zero feature ranges)
  - Int ID summary (species, moves, ability, item)
  - Aggregate stats: value distribution, entropy, action patterns, fallback rate

Saves:
  - Per-turn observations to diag_output/obs_dump.npz
  - Aggregate stats to diag_output/diag_summary.json

Usage:
    /c/Windows/py.exe -3 scripts/diagnose_obs.py --checkpoint checkpoints/ppo_latest.pkl --games 3
    /c/Windows/py.exe -3 scripts/diagnose_obs.py --checkpoint checkpoints/ppo_best.pkl --games 5 --save-obs
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
import numpy as np

sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1, closefd=False)


async def main():
    _tag = random.randint(1000, 9999)
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ppo_latest.pkl")
    parser.add_argument("--gen", type=int, default=4)
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--format", type=str, default="gen4randombattle")
    parser.add_argument("--server", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--save-obs", action="store_true",
                        help="Save raw observations to numpy files for offline analysis")
    parser.add_argument("--output-dir", type=str, default="diag_output")
    args = parser.parse_args()

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    import jax.numpy as jnp
    from poke_env import LocalhostServerConfiguration, AccountConfiguration
    from poke_env.player import SimpleHeuristicsPlayer
    from pokejax.players.showdown_player import (
        PokejaxPlayer, ObsBridge, N_TOKENS, FLOAT_DIM, N_ACTIONS, INT_IDS_PER_TOKEN,
        _OFF_HP_FRAC, _OFF_HP_BIN, _OFF_BASE_STATS, _OFF_BOOSTS, _OFF_STATUS,
        _OFF_VOLATILE, _OFF_TYPE1, _OFF_TYPE2, _OFF_IS_FAINTED, _OFF_IS_ACTIVE,
        _OFF_SLOT, _OFF_IS_OWN, _OFF_MOVES, _OFF_SLEEP_BIN, _OFF_REST_BIN,
        _OFF_SUB_FRAC, _OFF_FORCE_TRAP, _OFF_MOV_DIS, _OFF_CONF_BIN,
        _OFF_TAUNT, _OFF_ENCORE, _OFF_YAWN, _OFF_LEVEL, _OFF_PERISH_BIN,
        _OFF_PROTECT, _OFF_LOCKED_MOV,
    )

    server_config = LocalhostServerConfiguration

    # Patch to diagnose
    class DiagnosticPlayer(PokejaxPlayer):
        def __init__(self, save_obs=False, out_dir="diag_output", **kwargs):
            super().__init__(**kwargs)
            self._diag_turns = 0
            self._diag_anomalies = 0
            self._save_obs = save_obs
            self._out_dir = out_dir

            # Aggregate tracking
            self._all_values = []          # model value estimates per turn
            self._all_entropies = []       # policy entropy per turn
            self._all_max_probs = []       # max prob (confidence) per turn
            self._action_type_counts = {"move": 0, "switch": 0, "fallback": 0}
            self._move_action_counts = np.zeros(10, dtype=int)  # action histogram
            self._game_values = []         # (game_idx, turn, value, result)
            self._current_game = 0
            self._value_by_result = {"WIN": [], "LOSS": []}
            self._obs_list = []            # for saving
            self._uniform_dist_turns = 0   # near-uniform policy count
            self._species_id_zero_count = 0  # times active species_id=0 (unmapped)

        def _choose_move_impl(self, battle):
            """Instrumented version with full diagnostics."""
            available_moves = battle.available_moves
            available_switches = battle.available_switches
            trapped = battle.trapped if hasattr(battle, 'trapped') else False
            active = battle.active_pokemon

            obs = self.obs_bridge.build_obs(battle)

            if trapped:
                obs["legal_mask"][4:] = 0.0
                if obs["legal_mask"][:4].sum() == 0 and available_moves:
                    available_move_names = set(m.id for m in available_moves)
                    for i, m in enumerate(self.obs_bridge._last_own_move_list[:4]):
                        if m is not None and m.id in available_move_names:
                            obs["legal_mask"][i] = 1.0
                if obs["legal_mask"].sum() == 0:
                    obs["legal_mask"][0] = 1.0

            int_ids = jnp.array(obs["int_ids"])
            float_feats = jnp.array(obs["float_feats"])
            legal_mask = jnp.array(obs["legal_mask"])

            log_probs, value = self._forward(self.params, int_ids, float_feats, legal_mask)
            log_probs = np.array(log_probs)
            probs = np.exp(log_probs)
            masked_probs = probs * np.array(obs["legal_mask"])
            action = int(np.argmax(masked_probs))

            self._diag_turns += 1

            # --- AGGREGATE TRACKING ---
            val = float(value)
            self._all_values.append(val)
            self._all_max_probs.append(float(np.max(masked_probs)))

            legal_arr_agg = np.array(obs["legal_mask"])
            legal_probs_agg = masked_probs[legal_arr_agg > 0]
            if len(legal_probs_agg) > 1:
                lp = legal_probs_agg / (legal_probs_agg.sum() + 1e-8)
                ent = float(-np.sum(lp * np.log(lp + 1e-8)))
                max_ent = float(np.log(len(lp)))
                self._all_entropies.append(ent / max_ent if max_ent > 0 else 0)
            else:
                self._all_entropies.append(0.0)

            self._move_action_counts[action] += 1
            self._game_values.append((self._current_game, battle.turn, val))

            # Save obs for offline analysis
            if self._save_obs:
                self._obs_list.append({
                    "int_ids": obs["int_ids"].copy(),
                    "float_feats": obs["float_feats"].copy(),
                    "legal_mask": obs["legal_mask"].copy(),
                    "log_probs": log_probs.copy(),
                    "value": val,
                    "action": action,
                    "turn": battle.turn,
                    "game": self._current_game,
                })

            # Check species_id of active token
            active_token_check = None
            for t in range(1, 7):
                if obs["float_feats"][t][_OFF_IS_ACTIVE] > 0.5:
                    active_token_check = t
                    break
            if active_token_check and obs["int_ids"][active_token_check][0] == 0:
                self._species_id_zero_count += 1

            # --- DIAGNOSTICS ---
            own_team = self.obs_bridge._last_own_team
            own_move_list = self.obs_bridge._last_own_move_list
            trap_str = " [TRAPPED]" if trapped else ""

            print(f"\n{'='*70}")
            print(f"Turn {battle.turn} | Active: {active.species} "
                  f"(HP: {active.current_hp}/{active.max_hp}){trap_str}")

            # Available moves from poke-env
            move_names = [m.id for m in available_moves]
            switch_names = [p.species for p in available_switches]
            print(f"  poke-env available_moves: {move_names}")
            print(f"  poke-env available_switches: {switch_names}")

            # Own move list from bridge
            bridge_moves = [m.id if m else "None" for m in own_move_list[:4]]
            print(f"  bridge own_move_list: {bridge_moves}")

            # Check: do bridge moves match available moves?
            avail_set = set(move_names)
            bridge_set = set(m.id for m in own_move_list[:4] if m)
            move_match = avail_set.issubset(bridge_set)
            if not move_match:
                print(f"  [WARN] Move mismatch! available={avail_set} bridge={bridge_set}")
                self._diag_anomalies += 1

            # Legal mask
            legal_arr = np.array(obs["legal_mask"])
            legal_str = " ".join(f"{v:.0f}" for v in legal_arr)
            print(f"  legal_mask: [{legal_str}]")

            # Check: legal mask vs poke-env
            for i, m in enumerate(own_move_list[:4]):
                if m and m.id in avail_set and legal_arr[i] == 0:
                    print(f"  [WARN] Move {m.id} available but not legal in mask (slot {i})")
                    self._diag_anomalies += 1

            # Model output
            val = float(value)
            print(f"  model value: {val:.4f}")
            if abs(val) < 0.01:
                print(f"  [NOTE] Value near zero — model uncertain")

            # Action probabilities (sorted)
            print(f"  action probabilities:")
            sorted_actions = sorted(range(N_ACTIONS), key=lambda a: -probs[a])
            for a in sorted_actions:
                if probs[a] < 0.01 and a != action:
                    continue
                legal_tag = "L" if legal_arr[a] > 0 else " "
                chosen_tag = " <--" if a == action else ""
                if a < 4:
                    name = own_move_list[a].id if a < len(own_move_list) and own_move_list[a] else f"move{a}"
                else:
                    slot = a - 4
                    name = own_team[slot].species if slot < len(own_team) and own_team[slot] else f"slot{slot}"
                print(f"    [{legal_tag}] {a:>2d} ({name:>20s}): {probs[a]*100:6.2f}%{chosen_tag}")

            # Check entropy of legal action distribution
            legal_probs = masked_probs[legal_arr > 0]
            if len(legal_probs) > 1:
                legal_probs = legal_probs / (legal_probs.sum() + 1e-8)
                entropy = -np.sum(legal_probs * np.log(legal_probs + 1e-8))
                max_entropy = np.log(len(legal_probs))
                print(f"  entropy: {entropy:.3f} / {max_entropy:.3f} "
                      f"({entropy/max_entropy*100:.0f}% of max)")
                if entropy / max_entropy > 0.9:
                    print(f"  [WARN] Near-uniform distribution — model may be confused")
                    self._diag_anomalies += 1

            # Find which token index is the active Pokemon
            active_token = None
            for t in range(1, 7):  # tokens 1-6 = own team
                ff = np.array(obs["float_feats"][t])
                if ff[_OFF_IS_ACTIVE] > 0.5:
                    active_token = t
                    break

            if active_token is None:
                print(f"  [BUG] No token has is_active=1!")
                self._diag_anomalies += 1
                active_token = 1  # fallback

            own_active_float = np.array(obs["float_feats"][active_token])
            own_active_int = np.array(obs["int_ids"][active_token])
            print(f"  active token index: {active_token} (slot {active_token - 1})")
            print(f"  own active float features:")
            print(f"    hp_frac={own_active_float[_OFF_HP_FRAC]:.3f} "
                  f"(poke-env: {active.current_hp}/{active.max_hp})")
            print(f"    level={own_active_float[_OFF_LEVEL]:.2f} "
                  f"(poke-env: {active.level})")
            print(f"    is_active={own_active_float[_OFF_IS_ACTIVE]:.0f} "
                  f"is_own={own_active_float[_OFF_IS_OWN]:.0f} "
                  f"is_fainted={own_active_float[_OFF_IS_FAINTED]:.0f}")

            # Verify HP encoding
            expected_hp_frac = (active.current_hp or 0) / max(active.max_hp or 1, 1)
            if abs(own_active_float[_OFF_HP_FRAC] - expected_hp_frac) > 0.01:
                print(f"    [BUG] HP mismatch! encoded={own_active_float[_OFF_HP_FRAC]:.3f} "
                      f"expected={expected_hp_frac:.3f}")
                self._diag_anomalies += 1

            # Verify base stats
            bst = getattr(active, 'base_stats', {}) or {}
            stat_order = ['hp', 'atk', 'def', 'spa', 'spd', 'spe']
            encoded_bst = own_active_float[_OFF_BASE_STATS:_OFF_BASE_STATS+6]
            bst_str = " ".join(f"{s}={encoded_bst[i]*255:.0f}" for i, s in enumerate(stat_order))
            print(f"    base_stats: {bst_str}")

            # Check boosts
            boosts = dict(active.boosts) if active.boosts else {}
            boost_order = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion']
            boost_strs = []
            for i, bname in enumerate(boost_order):
                encoded = own_active_float[_OFF_BOOSTS + i*13:_OFF_BOOSTS + (i+1)*13]
                enc_idx = int(np.argmax(encoded)) - 6 if encoded.max() > 0 else 0
                actual = boosts.get(bname, 0)
                if enc_idx != actual:
                    boost_strs.append(f"{bname}:enc={enc_idx}/actual={actual}")
                elif actual != 0:
                    boost_strs.append(f"{bname}={actual}")
            if boost_strs:
                print(f"    boosts: {', '.join(boost_strs)}")

            # Check status encoding
            status_vec = own_active_float[_OFF_STATUS:_OFF_STATUS+7]
            status_idx = int(np.argmax(status_vec)) if status_vec.max() > 0 else 0
            status_names = ["none", "burn", "poison", "toxic", "sleep", "freeze", "paralyze"]
            print(f"    status={status_names[status_idx]} (poke-env: {active.status})")

            # Check volatile encoding
            vol_vec = own_active_float[_OFF_VOLATILE:_OFF_VOLATILE+27]
            active_vols = np.where(vol_vec > 0)[0]
            vol_name_map = {v: k for k, v in {
                "confusion": 0, "infatuation": 1, "leechseed": 2,
                "curse": 3, "aquaring": 4, "ingrain": 5,
                "taunt": 6, "encore": 7, "flinch": 8,
                "embargo": 9, "healblock": 10, "magnetrise": 11,
                "partiallytrapped": 12, "perishsong": 13,
                "powertrick": 14, "substitute": 15, "yawn": 16,
                "focusenergy": 17, "charge": 18, "stockpile": 19,
                "torment": 20, "nightmare": 21, "imprison": 22,
                "mustrecharge": 23, "twoturnmove": 24,
                "destinybond": 25, "grudge": 26,
            }.items()}
            if len(active_vols):
                vol_names = [vol_name_map.get(v, f"?{v}") for v in active_vols]
                print(f"    volatiles: {vol_names}")

            # Check move features in active token + PP tracking
            for mi in range(4):
                moff = _OFF_MOVES + mi * 45
                mfeats = own_active_float[moff:moff+45]
                known = mfeats[44]  # is_known flag
                pp_frac_encoded = mfeats[43]  # pp_frac in observation
                if known > 0.5:
                    type_vec = mfeats[14:32]
                    type_idx = int(np.argmax(type_vec)) if type_vec.max() > 0 else -1
                    cat_vec = mfeats[32:35]
                    cat_idx = int(np.argmax(cat_vec)) if cat_vec.max() > 0 else -1
                    cat_names = ["physical", "special", "status"]
                    cat = cat_names[cat_idx] if 0 <= cat_idx < 3 else "?"
                    m = own_move_list[mi] if mi < len(own_move_list) else None
                    move_name = m.id if m else "?"
                    real_pp = m.current_pp if m and m.current_pp is not None else "?"
                    real_maxpp = m.max_pp if m and m.max_pp else "?"
                    print(f"    move{mi}: {move_name} type={type_idx} "
                          f"cat={cat} pp_enc={pp_frac_encoded:.3f} "
                          f"real_pp={real_pp}/{real_maxpp}")
                    # Check available_moves PP too
                    for am in available_moves:
                        if am.id == move_name:
                            print(f"            avail_move pp: {am.current_pp}/{am.max_pp}")
                            break

            # Team summary — show all 6 own slots
            print(f"  own team slots:")
            for t in range(1, 7):
                ff = np.array(obs["float_feats"][t])
                ii = np.array(obs["int_ids"][t])
                hp = ff[_OFF_HP_FRAC]
                is_act = ff[_OFF_IS_ACTIVE]
                is_faint = ff[_OFF_IS_FAINTED]
                slot_p = own_team[t-1]
                name = slot_p.species if slot_p else "empty"
                tag = " *ACTIVE*" if is_act > 0.5 else ""
                tag += " FAINTED" if is_faint > 0.5 else ""
                print(f"    slot{t-1}: {name:>15s} hp={hp:.3f} species_id={ii[0]}{tag}")

            # Opponent token summary + type effectiveness of chosen move
            opp_with_data = 0
            opp_active_type_indices = []
            print(f"  opp team slots:")
            for t in range(7, 13):
                ff = np.array(obs["float_feats"][t])
                ii = np.array(obs["int_ids"][t])
                nz = np.count_nonzero(ff)
                if nz > 0:
                    opp_with_data += 1
                    hp = ff[_OFF_HP_FRAC]
                    is_act = ff[_OFF_IS_ACTIVE]
                    lvl = ff[_OFF_LEVEL]
                    # Extract opponent types
                    type1_vec = ff[_OFF_TYPE1:_OFF_TYPE1+18]
                    type2_vec = ff[_OFF_TYPE2:_OFF_TYPE2+18]
                    t1 = int(np.argmax(type1_vec)) if type1_vec.max() > 0 else -1
                    t2 = int(np.argmax(type2_vec)) if type2_vec.max() > 0 else -1
                    type_names = ["Normal","Fire","Water","Electric","Grass","Ice",
                                  "Fighting","Poison","Ground","Flying","Psychic","Bug",
                                  "Rock","Ghost","Dragon","Dark","Steel","Fairy"]
                    t1n = type_names[t1] if 0 <= t1 < 18 else "?"
                    t2n = type_names[t2] if 0 <= t2 < 18 else ""
                    type_str = f"{t1n}/{t2n}" if t2n else t1n
                    if is_act > 0.5:
                        opp_active_type_indices = [t1]
                        if t2 >= 0 and type2_vec.max() > 0:
                            opp_active_type_indices.append(t2)
                    print(f"    slot{t-7}: species_id={ii[0]} hp={hp:.3f} "
                          f"is_active={is_act:.0f} level={lvl:.2f} "
                          f"types={type_str} nonzero={nz}")
            print(f"  opp tokens with data: {opp_with_data}/6")

            # Check chosen move's type vs opponent's types (basic effectiveness)
            if action < 4 and opp_active_type_indices:
                mi = action
                moff = _OFF_MOVES + mi * 45
                mfeats = own_active_float[moff:moff+45]
                move_type_vec = mfeats[14:32]
                move_type_idx = int(np.argmax(move_type_vec)) if move_type_vec.max() > 0 else -1
                if move_type_idx >= 0:
                    type_names_short = ["Normal","Fire","Water","Elec","Grass","Ice",
                                        "Fight","Poison","Ground","Fly","Psychic","Bug",
                                        "Rock","Ghost","Dragon","Dark","Steel","Fairy"]
                    mtn = type_names_short[move_type_idx] if move_type_idx < 18 else "?"
                    opp_tn = [type_names_short[t] for t in opp_active_type_indices if t < 18]
                    print(f"  type matchup: {mtn} → {'/'.join(opp_tn)}")

            # ---- Actually execute the action ----
            own_team = self.obs_bridge._last_own_team
            own_move_list = self.obs_bridge._last_own_move_list

            if action < 4:
                if action < len(own_move_list) and own_move_list[action] is not None:
                    chosen_move_id = own_move_list[action].id
                    for m in available_moves:
                        if m.id == chosen_move_id:
                            print(f"  -> CHOSE: move {chosen_move_id}")
                            self._action_type_counts["move"] += 1
                            return self.create_order(m)
                print(f"  [WARN] Move action {action} fell through to fallback")
                self._diag_anomalies += 1
                self._action_type_counts["fallback"] += 1
                if available_moves:
                    return self.create_order(available_moves[0])
            else:
                slot = action - 4
                if slot < len(own_team) and own_team[slot] is not None:
                    target = own_team[slot]
                    if target is active:
                        print(f"  [BUG] Switch to active {active.species}!")
                        self._diag_anomalies += 1
                        self._action_type_counts["fallback"] += 1
                    elif target in available_switches:
                        print(f"  -> CHOSE: switch to {target.species}")
                        self._action_type_counts["switch"] += 1
                        return self.create_order(target)
                    else:
                        print(f"  [WARN] Switch target {target.species} not in available_switches")
                        self._diag_anomalies += 1
                        self._action_type_counts["fallback"] += 1
                if available_switches:
                    return self.create_order(available_switches[0])

            self._action_type_counts["fallback"] += 1
            if available_moves:
                return self.create_order(available_moves[0])
            if available_switches:
                return self.create_order(available_switches[0])
            return self.choose_default_move()

    # Create players
    bot = DiagnosticPlayer(
        checkpoint_path=args.checkpoint,
        gen=args.gen,
        verbose=False,  # we handle our own logging
        save_obs=args.save_obs,
        out_dir=out_dir,
        account_configuration=AccountConfiguration(f"DiagBot{_tag}", None),
        server_configuration=server_config,
        battle_format=args.format,
        max_concurrent_battles=1,
    )
    opp = SimpleHeuristicsPlayer(
        account_configuration=AccountConfiguration(f"HeurBot{_tag}", None),
        server_configuration=server_config,
        battle_format=args.format,
        max_concurrent_battles=1,
    )

    wins = 0
    losses = 0
    game_results = []
    for game in range(1, args.games + 1):
        bot._current_game = game
        await bot.battle_against(opp, n_battles=1)
        n_wins = bot.n_won_battles
        n_total = bot.n_finished_battles
        result = "WIN" if n_wins > wins else "LOSS"
        if result == "WIN":
            wins += 1
        else:
            losses += 1
        game_results.append(result)

        # Track values by result
        game_vals = [v for g, t, v in bot._game_values if g == game]
        if game_vals:
            bot._value_by_result[result].extend(game_vals)

        print(f"\n{'#'*70}")
        print(f"Game {game}: {result} | {wins}W/{losses}L ({wins/n_total*100:.0f}%)")
        print(f"Anomalies this session: {bot._diag_anomalies}")
        print(f"Total turns: {bot._diag_turns}")
        print(f"{'#'*70}\n")

    # ====================================================================
    # AGGREGATE SUMMARY
    # ====================================================================
    print(f"\n{'='*70}")
    print(f"AGGREGATE DIAGNOSTIC SUMMARY")
    print(f"{'='*70}")
    print(f"\nResults: {wins}W/{losses}L ({wins/(wins+losses)*100:.1f}%)")
    print(f"Total turns: {bot._diag_turns}")
    print(f"Total anomalies: {bot._diag_anomalies}")
    print(f"Species ID = 0 (unmapped) on active: {bot._species_id_zero_count}")

    # Value estimates
    vals = np.array(bot._all_values)
    if len(vals) > 0:
        print(f"\nValue estimates:")
        print(f"  mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"min={vals.min():.4f}  max={vals.max():.4f}")
        print(f"  median={np.median(vals):.4f}")
        # Value by result
        for res in ["WIN", "LOSS"]:
            rv = bot._value_by_result.get(res, [])
            if rv:
                rv = np.array(rv)
                print(f"  {res}: mean={rv.mean():.4f}  std={rv.std():.4f}")

    # Entropy
    ents = np.array(bot._all_entropies)
    if len(ents) > 0:
        print(f"\nPolicy entropy (fraction of max):")
        print(f"  mean={ents.mean():.3f}  std={ents.std():.3f}  "
              f"min={ents.min():.3f}  max={ents.max():.3f}")
        near_uniform = (ents > 0.9).sum()
        print(f"  near-uniform (>0.9): {near_uniform}/{len(ents)} "
              f"({near_uniform/len(ents)*100:.1f}%)")

    # Confidence
    maxp = np.array(bot._all_max_probs)
    if len(maxp) > 0:
        print(f"\nMax probability (confidence):")
        print(f"  mean={maxp.mean():.3f}  std={maxp.std():.3f}  "
              f"min={maxp.min():.3f}  max={maxp.max():.3f}")

    # Action distribution
    print(f"\nAction type counts:")
    for k, v in bot._action_type_counts.items():
        pct = v / max(bot._diag_turns, 1) * 100
        print(f"  {k}: {v} ({pct:.1f}%)")
    print(f"\nAction histogram (0-3=moves, 4-9=switches):")
    for a in range(10):
        cnt = bot._move_action_counts[a]
        if cnt > 0:
            pct = cnt / max(bot._diag_turns, 1) * 100
            print(f"  action {a}: {cnt} ({pct:.1f}%)")

    # Save obs dumps
    if args.save_obs and bot._obs_list:
        obs_path = os.path.join(out_dir, "obs_dump.npz")
        np.savez_compressed(
            obs_path,
            int_ids=np.array([o["int_ids"] for o in bot._obs_list]),
            float_feats=np.array([o["float_feats"] for o in bot._obs_list]),
            legal_masks=np.array([o["legal_mask"] for o in bot._obs_list]),
            log_probs=np.array([o["log_probs"] for o in bot._obs_list]),
            values=np.array([o["value"] for o in bot._obs_list]),
            actions=np.array([o["action"] for o in bot._obs_list]),
            turns=np.array([o["turn"] for o in bot._obs_list]),
            games=np.array([o["game"] for o in bot._obs_list]),
        )
        print(f"\nSaved {len(bot._obs_list)} observations to {obs_path}")

    # Save summary
    summary = {
        "checkpoint": args.checkpoint,
        "games": args.games,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / (wins + losses),
        "total_turns": bot._diag_turns,
        "anomalies": bot._diag_anomalies,
        "species_id_zero_count": bot._species_id_zero_count,
        "value_mean": float(vals.mean()) if len(vals) else 0,
        "value_std": float(vals.std()) if len(vals) else 0,
        "entropy_mean": float(ents.mean()) if len(ents) else 0,
        "near_uniform_frac": float(near_uniform / len(ents)) if len(ents) else 0,
        "confidence_mean": float(maxp.mean()) if len(maxp) else 0,
        "action_types": bot._action_type_counts,
        "fallback_rate": bot._action_type_counts["fallback"] / max(bot._diag_turns, 1),
        "game_results": game_results,
    }
    summary_path = os.path.join(out_dir, "diag_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
