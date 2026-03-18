#!/usr/bin/env python3
"""
Diagnostic script: Compare training obs_builder vs PS ObsBridge encodings.

Runs battles on the local PS server, captures poke-env Battle objects,
converts them BOTH ways (training obs_builder via BattleBridge → engine state
→ build_obs, AND PS ObsBridge.build_obs), and reports differences.

This isolates observation encoding mismatches that cause the model to
see different inputs at inference vs training time.

Prerequisites:
    Start local Showdown server:
        cd pokemon-showdown && node pokemon-showdown start --no-security

Usage:
    python scripts/diagnose_obs_mismatch.py --games 5 --verbose
"""

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout = open(sys.stdout.fileno(), "w", buffering=1, closefd=False)


# Feature offset names for human-readable diff output
_FEATURE_NAMES = {
    0: "hp_frac",
    1: "hp_bin[0:10]",
    11: "base_stats[0:6]",
    17: "boosts[0:91]",
    108: "status[0:7]",
    115: "volatile[0:27]",
    142: "type1[0:18]",
    160: "type2[0:18]",
    178: "is_fainted",
    179: "is_active",
    180: "slot[0:6]",
    186: "is_own",
    187: "moves[0:180]",
    367: "sleep_bin[0:4]",
    371: "rest_bin[0:3]",
    374: "sub_frac",
    375: "force_trap",
    376: "mov_dis[0:4]",
    380: "conf_bin[0:4]",
    384: "taunt",
    385: "encore",
    386: "yawn",
    387: "level",
    388: "perish_bin[0:4]",
    392: "protect",
    393: "locked_mov",
}

# Field feature offset names
_FIELD_NAMES = {
    0: "weather[0:5]",
    5: "wt_turns[0:8]",
    13: "pseudo[0:5]",
    18: "tr_turns[0:4]",
    22: "hazards_own[0:7]",
    29: "hazards_opp[0:7]",
    36: "screens_own[0:6]",
    42: "screens_opp[0:6]",
    48: "turn_bin[0:10]",
    58: "fainted[0:2]",
    60: "toxic_own[0:5]",
    65: "toxic_opp[0:5]",
    70: "tailwind[0:2]",
    72: "wish[0:2]",
    74: "safeguard[0:2]",
    76: "mist[0:2]",
    78: "lucky_chant[0:2]",
    80: "gravity_t[0:4]",
}


def _get_feature_name(offset, is_field=False):
    """Get human-readable feature name for a given offset."""
    names = _FIELD_NAMES if is_field else _FEATURE_NAMES
    best_name = f"offset_{offset}"
    best_off = -1
    for off, name in sorted(names.items()):
        if off <= offset and off > best_off:
            best_name = name
            best_off = off
            if "[" in name:
                idx = offset - off
                base = name.split("[")[0]
                best_name = f"{base}[{idx}]"
            elif off == offset:
                best_name = name
    return best_name


def compare_obs(obs_ps, obs_engine, turn, verbose=False):
    """Compare two observation dicts and report differences."""
    diffs = []

    # Compare int_ids
    int_ps = obs_ps["int_ids"]
    int_eng = obs_engine["int_ids"]
    int_diff_mask = int_ps != int_eng
    if int_diff_mask.any():
        for t in range(15):
            for f in range(8):
                if int_diff_mask[t, f]:
                    token_name = _token_name(t)
                    diffs.append({
                        "type": "int_ids",
                        "token": t,
                        "token_name": token_name,
                        "field": f,
                        "ps_val": int(int_ps[t, f]),
                        "eng_val": int(int_eng[t, f]),
                    })

    # Compare float_feats
    float_ps = obs_ps["float_feats"]
    float_eng = obs_engine["float_feats"]
    float_diff = np.abs(float_ps - float_eng)
    threshold = 0.01
    diff_mask = float_diff > threshold

    if diff_mask.any():
        for t in range(15):
            for f in range(394):
                if diff_mask[t, f]:
                    token_name = _token_name(t)
                    is_field = (t == 0)
                    feat_name = _get_feature_name(f, is_field=is_field)
                    diffs.append({
                        "type": "float_feats",
                        "token": t,
                        "token_name": token_name,
                        "offset": f,
                        "feat_name": feat_name,
                        "ps_val": float(float_ps[t, f]),
                        "eng_val": float(float_eng[t, f]),
                        "diff": float(float_diff[t, f]),
                    })

    # Compare legal_mask
    mask_ps = obs_ps["legal_mask"]
    mask_eng = obs_engine["legal_mask"]
    mask_diff = np.abs(mask_ps - mask_eng) > 0.01
    if mask_diff.any():
        for a in range(10):
            if mask_diff[a]:
                action_name = f"move_{a}" if a < 4 else f"switch_{a-4}"
                diffs.append({
                    "type": "legal_mask",
                    "action": a,
                    "action_name": action_name,
                    "ps_val": float(mask_ps[a]),
                    "eng_val": float(mask_eng[a]),
                })

    return diffs


def _token_name(t):
    if t == 0:
        return "FIELD"
    elif 1 <= t <= 6:
        return f"OWN_TEAM[{t-1}]"
    elif 7 <= t <= 12:
        return f"OPP_TEAM[{t-7}]"
    elif t == 13:
        return "ACTOR"
    elif t == 14:
        return "CRITIC"
    return f"TOKEN_{t}"


class DiagnosticPlayer:
    """Captures poke-env Battle objects and compares obs encodings."""

    def __init__(self, obs_bridge, battle_bridge, tables, log):
        self.obs_bridge = obs_bridge
        self.battle_bridge = battle_bridge
        self.tables = tables
        self.log = log
        self.all_diffs = []
        self.diff_summary = {}  # feat_name → count

    def diagnose_battle(self, battle, turn):
        """Compare obs encodings for a single battle turn."""
        import jax
        import jax.numpy as jnp
        from pokejax.rl.obs_builder import build_obs as engine_build_obs

        # Method 1: PS ObsBridge (what the model sees at inference)
        obs_ps = self.obs_bridge.build_obs(battle)

        # Method 2: BattleBridge → engine state → training obs_builder
        rng_key = jax.random.PRNGKey(42)
        try:
            state, reveal = self.battle_bridge.battle_to_state(battle, rng_key)
            obs_engine = engine_build_obs(state, reveal, 0, self.tables)
            # Convert JAX arrays to numpy
            obs_engine_np = {
                "int_ids": np.array(obs_engine["int_ids"]),
                "float_feats": np.array(obs_engine["float_feats"]),
                "legal_mask": np.array(obs_engine["legal_mask"]),
            }
        except Exception as e:
            self.log.warning(f"  BattleBridge conversion failed at turn {turn}: {e}")
            return

        diffs = compare_obs(obs_ps, obs_engine_np, turn)

        if diffs:
            self.all_diffs.extend(diffs)
            for d in diffs:
                key = d.get("feat_name", d.get("action_name", d["type"]))
                self.diff_summary[key] = self.diff_summary.get(key, 0) + 1

        return diffs

    def print_summary(self):
        """Print summary of all differences found."""
        if not self.diff_summary:
            self.log.info("No observation encoding differences found!")
            return

        self.log.info("=" * 70)
        self.log.info("OBSERVATION ENCODING DIFFERENCES SUMMARY")
        self.log.info("=" * 70)

        # Sort by count (most frequent first)
        sorted_diffs = sorted(self.diff_summary.items(), key=lambda x: -x[1])
        for feat, count in sorted_diffs:
            self.log.info(f"  {feat:40s} : {count:5d} occurrences")

        self.log.info(f"\nTotal differences: {len(self.all_diffs)}")
        self.log.info(f"Unique features with differences: {len(self.diff_summary)}")


async def run_diagnostic(args, log):
    """Run diagnostic games and compare obs encodings."""
    import jax

    from poke_env import LocalhostServerConfiguration, AccountConfiguration
    from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer, Player
    from poke_env.environment import AbstractBattle

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pokejax.players.showdown_player import PokejaxPlayer, ObsBridge
    from pokejax.search.battle_bridge import BattleBridge
    from pokejax.data.tables import load_tables

    _tag = random.randint(1000, 9999)

    # Load tables
    tables = load_tables(args.gen)
    obs_bridge = ObsBridge(tables)
    battle_bridge = BattleBridge(obs_bridge)
    diag = DiagnosticPlayer(obs_bridge, battle_bridge, tables, log)

    # Create a diagnostic player that captures battles
    class DiagPlayer(PokejaxPlayer):
        def __init__(self, diag_obj, **kwargs):
            super().__init__(**kwargs)
            self.diag = diag_obj

        def choose_move(self, battle):
            # Run diagnostic comparison
            try:
                diffs = self.diag.diagnose_battle(battle, battle.turn)
                if diffs and args.verbose:
                    log.info(f"  Turn {battle.turn}: {len(diffs)} obs diffs found")
                    for d in diffs[:5]:  # Show first 5
                        if d["type"] == "float_feats":
                            log.info(f"    {d['token_name']}.{d['feat_name']}: "
                                     f"PS={d['ps_val']:.4f} vs Eng={d['eng_val']:.4f}")
                        elif d["type"] == "int_ids":
                            log.info(f"    {d['token_name']}.int[{d['field']}]: "
                                     f"PS={d['ps_val']} vs Eng={d['eng_val']}")
                        elif d["type"] == "legal_mask":
                            log.info(f"    {d['action_name']}: "
                                     f"PS={d['ps_val']:.0f} vs Eng={d['eng_val']:.0f}")
                    if len(diffs) > 5:
                        log.info(f"    ... and {len(diffs) - 5} more")
            except Exception as e:
                log.warning(f"  Diagnostic failed: {e}")

            # Still play normally
            return super().choose_move(battle)

    player = DiagPlayer(
        diag,
        checkpoint_path=args.checkpoint,
        gen=args.gen,
        temperature=0.0,
        verbose=args.verbose,
        account_configuration=AccountConfiguration(f"DiagBot{_tag}", None),
        server_configuration=LocalhostServerConfiguration,
        battle_format=args.format,
    )

    opponent = SimpleHeuristicsPlayer(
        account_configuration=AccountConfiguration(f"HeuristicBot{_tag}", None),
        server_configuration=LocalhostServerConfiguration,
        battle_format=args.format,
    )

    log.info(f"Running {args.games} diagnostic games...")
    await player.battle_against(opponent, n_battles=args.games)

    # Count results
    wins = sum(1 for b in player.battles.values() if b.won is True)
    losses = sum(1 for b in player.battles.values() if b.won is False)
    ties = sum(1 for b in player.battles.values() if b.won is None and b.finished)

    log.info(f"\nResults: {wins}W/{losses}L/{ties}T ({wins}/{wins+losses+ties} = "
             f"{wins/max(wins+losses+ties, 1)*100:.0f}%)")

    diag.print_summary()

    # Save detailed diffs
    out_path = Path(args.output_dir) / "obs_diffs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "summary": diag.diff_summary,
            "total_diffs": len(diag.all_diffs),
            "sample_diffs": diag.all_diffs[:200],  # First 200 for inspection
        }, f, indent=2)
    log.info(f"\nDetailed diffs saved to {out_path}")


async def main():
    parser = argparse.ArgumentParser(
        description="Diagnose obs encoding differences between training and PS"
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ppo_best.pkl")
    parser.add_argument("--gen", type=int, default=4)
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--format", type=str, default="gen4randombattle")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-dir", type=str, default="diagnostic_results")
    args = parser.parse_args()

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    log = logging.getLogger("diagnose")

    await run_diagnostic(args, log)


if __name__ == "__main__":
    asyncio.run(main())
