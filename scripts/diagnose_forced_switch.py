#!/usr/bin/env python3
"""
Diagnostic script: log model behavior in the pokejax engine to check
forced switch handling and compare with PS behavior.

Runs N battles in the JAX engine with verbose logging of:
  - Every action the model picks
  - Every forced switch (auto-handled by the engine)
  - Action probabilities when the model would face a "force switch only" mask
  - Double-switch patterns (switch immediately after a forced switch)

Usage (WSL):
    python3 scripts/diagnose_forced_switch.py --checkpoint checkpoints/ppo_best.pkl --games 20
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.config import GenConfig
from pokejax.data.tables import load_tables
from pokejax.env.pokejax_env import PokeJAXEnv, EnvState
from pokejax.rl.obs_builder import build_obs
from pokejax.rl.model import create_model
from pokejax.engine.switch import get_valid_switch_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ppo_best.pkl")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--gen", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=300)
    args = parser.parse_args()

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    with open(args.checkpoint, "rb") as f:
        ckpt = pickle.load(f)
    arch = ckpt.get("arch", "transformer")
    model = create_model(arch)
    params = ckpt["params"]

    @jax.jit
    def forward(params, int_ids, float_feats, legal_mask):
        log_probs, _, value = model.apply(
            params, int_ids[None], float_feats[None], legal_mask[None],
        )
        return log_probs[0], value[0]

    # Warm up JIT
    dummy_int = jnp.zeros((15, 8), dtype=jnp.int32)
    dummy_float = jnp.zeros((15, 394), dtype=jnp.float32)
    dummy_mask = jnp.ones(10, dtype=jnp.float32)
    _ = forward(params, dummy_int, dummy_float, dummy_mask)
    print("JIT warmup done.")

    # Load env
    env = PokeJAXEnv(gen=args.gen)
    tables = env.tables

    # Heuristic opponent: pick highest-damage move or random switch
    def heuristic_action(state, side, key):
        """Simple heuristic: pick highest base power move, or random switch."""
        from pokejax.env.action_mask import get_action_mask
        mask = get_action_mask(state, side)
        move_mask = mask[:4]
        switch_mask = mask[4:]

        if move_mask.any():
            # Pick highest BP move among legal
            idx = state.sides_active_idx[side]
            move_ids = state.sides_team_move_ids[side, idx]
            bps = jnp.array([
                jnp.where(move_ids[i] >= 0, tables.moves[jnp.clip(move_ids[i], 0, len(tables.moves)-1), 0], 0)
                for i in range(4)
            ])
            bps = jnp.where(move_mask, bps, -1)
            return jnp.argmax(bps).astype(jnp.int32)
        elif switch_mask.any():
            # Random switch among legal
            probs = switch_mask.astype(jnp.float32)
            probs = probs / probs.sum()
            return jax.random.categorical(key, jnp.log(probs + 1e-8)) + 4
        else:
            return jnp.int32(0)

    # Stats
    total_wins = 0
    total_losses = 0
    total_forced_switches = 0
    total_double_switches = 0
    total_turns = 0

    # Simulate what the model would pick for forced-switch-only masks
    forced_switch_choices = []  # (chosen_slot, prob_of_choice)

    key = jax.random.PRNGKey(42)

    for game_idx in range(args.games):
        key, reset_key = jax.random.split(key)
        env_state, _ = env.reset(reset_key)
        state = env_state.battle

        last_action_was_forced_switch = False
        game_turns = 0

        print(f"\n{'='*60}")
        print(f"Game {game_idx+1}/{args.games}")
        print(f"{'='*60}")

        while not state.finished and game_turns < args.max_turns:
            game_turns += 1

            # Build obs for player 0 (model)
            obs = build_obs(state, env_state.reveal, player=0, tables=tables)

            # Forward pass
            log_probs, value = forward(
                params,
                obs["int_ids"], obs["float_feats"], obs["legal_mask"],
            )
            probs = np.exp(np.array(log_probs))
            legal = np.array(obs["legal_mask"])

            # Pick action (greedy)
            masked_probs = probs * legal
            action_p0 = int(np.argmax(masked_probs))

            # Log the action
            active_idx_0 = int(state.sides_active_idx[0])
            active_species_0 = int(state.sides_team_species_id[0, active_idx_0])
            active_hp_0 = int(state.sides_team_hp[0, active_idx_0])
            active_maxhp_0 = int(state.sides_team_max_hp[0, active_idx_0])

            if action_p0 < 4:
                move_id = int(state.sides_team_move_ids[0, active_idx_0, action_p0])
                print(f"  T{game_turns}: Move {action_p0} (id={move_id}) "
                      f"prob={masked_probs[action_p0]*100:.1f}% "
                      f"v={float(value):.3f} "
                      f"[species={active_species_0} hp={active_hp_0}/{active_maxhp_0}]")
            else:
                slot = action_p0 - 4
                switch_species = int(state.sides_team_species_id[0, slot])
                switch_hp = int(state.sides_team_hp[0, slot])
                is_double_switch = last_action_was_forced_switch
                ds_marker = " ** DOUBLE SWITCH **" if is_double_switch else ""
                print(f"  T{game_turns}: Switch to slot {slot} (species={switch_species} "
                      f"hp={switch_hp}) "
                      f"prob={masked_probs[action_p0]*100:.1f}% "
                      f"v={float(value):.3f}{ds_marker}")
                if is_double_switch:
                    total_double_switches += 1

            # Also simulate what the model would do with a forced-switch-only mask
            forced_mask = np.zeros(10, dtype=np.float32)
            switch_valid = np.array(get_valid_switch_mask(state, 0))
            forced_mask[4:] = switch_valid.astype(np.float32)
            if forced_mask.sum() > 0:
                forced_log_probs, _ = forward(
                    params,
                    obs["int_ids"], obs["float_feats"],
                    jnp.array(forced_mask),
                )
                forced_probs = np.exp(np.array(forced_log_probs))
                forced_masked = forced_probs * forced_mask
                if forced_masked.sum() > 0:
                    forced_action = int(np.argmax(forced_masked))
                    forced_slot = forced_action - 4
                    forced_switch_choices.append({
                        "slot": forced_slot,
                        "prob": float(forced_masked[forced_action]),
                        "all_probs": forced_masked[4:].tolist(),
                    })

            # Opponent action (heuristic)
            key, opp_key = jax.random.split(key)
            action_p1 = heuristic_action(state, 1, opp_key)

            actions = jnp.array([action_p0, int(action_p1)], dtype=jnp.int32)

            # Step
            key, step_key = jax.random.split(key)

            # Check pre-step state for forced switch detection
            pre_active_0 = int(state.sides_active_idx[0])
            pre_fainted_0 = [bool(state.sides_team_fainted[0, i]) for i in range(6)]

            new_env_state, _, rewards, dones, _ = env.step(env_state, actions, step_key)

            # Detect forced switches (active changed without model choosing switch)
            post_active_0 = int(new_env_state.battle.sides_active_idx[0])
            post_fainted_0 = [bool(new_env_state.battle.sides_team_fainted[0, i]) for i in range(6)]

            new_faints = sum(1 for i in range(6) if post_fainted_0[i] and not pre_fainted_0[i])

            if post_active_0 != pre_active_0 and action_p0 < 4:
                # Active changed but model chose a move = forced switch happened
                total_forced_switches += 1
                new_species = int(new_env_state.battle.sides_team_species_id[0, post_active_0])
                print(f"  ** FORCED SWITCH: slot {pre_active_0} fainted -> "
                      f"auto-switched to slot {post_active_0} (species={new_species})")
                last_action_was_forced_switch = True
            elif action_p0 >= 4:
                last_action_was_forced_switch = False
            else:
                last_action_was_forced_switch = False

            env_state = new_env_state
            state = env_state.battle
            total_turns += 1

        # Game result
        if state.finished:
            winner = int(state.winner)
            if winner == 0:
                total_wins += 1
                print(f"  Result: WIN")
            else:
                total_losses += 1
                print(f"  Result: LOSS")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Games: {args.games}")
    print(f"Wins: {total_wins} ({total_wins/max(args.games,1)*100:.1f}%)")
    print(f"Losses: {total_losses}")
    print(f"Total turns: {total_turns}")
    print(f"Total forced switches (auto): {total_forced_switches}")
    print(f"Total double switches (switch right after forced): {total_double_switches}")
    print(f"Avg turns/game: {total_turns/max(args.games,1):.1f}")

    if forced_switch_choices:
        # Analyze forced switch distributions
        print(f"\n--- Forced Switch Distribution Analysis ---")
        print(f"Simulated forced-switch-only scenarios: {len(forced_switch_choices)}")
        avg_max_prob = np.mean([c["prob"] for c in forced_switch_choices])
        print(f"Avg probability of chosen slot: {avg_max_prob*100:.1f}%")

        # How often does the model pick slot 0 (lowest alive)?
        slot_counts = [0] * 6
        for c in forced_switch_choices:
            slot_counts[c["slot"]] = slot_counts[c["slot"]] + 1
        print(f"Slot distribution: {slot_counts}")
        print(f"(Training always picks lowest alive slot)")

        # Show a few example distributions
        print(f"\nExample forced-switch probability distributions:")
        for i, c in enumerate(forced_switch_choices[:5]):
            probs_str = " ".join(f"s{j}={p*100:.1f}%" for j, p in enumerate(c["all_probs"]) if p > 0)
            print(f"  #{i+1}: {probs_str} -> chose slot {c['slot']}")


if __name__ == "__main__":
    main()
