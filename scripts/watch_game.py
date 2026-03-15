#!/usr/bin/env python3
"""
Play and watch games between BC model and heuristic/random opponents.

Prints a human-readable battle log showing each turn's decisions,
HP changes, status effects, and game outcome.

Usage:
    python scripts/watch_game.py --checkpoint checkpoints/bc_final.pkl --vs heuristic --games 3
    python scripts/watch_game.py --checkpoint checkpoints/bc_final.pkl --vs random --games 5
"""

import argparse
import os
import sys
import pickle
import time

sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1, closefd=False)

import numpy as np
import jax
import jax.numpy as jnp


STATUS_NAMES = {0: "", 1: "BRN", 2: "PSN", 3: "TOX", 4: "SLP", 5: "FRZ", 6: "PAR"}
WEATHER_NAMES = {0: "", 1: "Sun", 2: "Rain", 3: "Sand", 4: "Hail"}
TYPE_NAMES = [
    "???", "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
    "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug",
    "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy",
]


def get_species_name(tables, species_id):
    sid = int(species_id)
    if 0 < sid < len(tables.species_names):
        return tables.species_names[sid]
    return f"???({sid})"


def get_move_name(tables, move_id):
    mid = int(move_id)
    if 0 < mid < len(tables.move_names):
        return tables.move_names[mid]
    return f"???({mid})"


def get_item_name(tables, item_id):
    iid = int(item_id)
    if 0 < iid < len(tables.item_names):
        return tables.item_names[iid]
    return ""


def get_ability_name(tables, ability_id):
    aid = int(ability_id)
    if 0 < aid < len(tables.ability_names):
        return tables.ability_names[aid]
    return ""


def format_action(action, state, side, tables):
    """Format an action (0-9) as human-readable text."""
    a = int(action)
    s = _to_np(state)
    active_idx = int(s['active_idx'][side])

    if a < 4:
        move_id = int(s['move_ids'][side, active_idx, a])
        move_name = get_move_name(tables, move_id)
        return f"uses {move_name}"
    else:
        slot = a - 4
        species_id = int(s['species_id'][side, slot])
        species_name = get_species_name(tables, species_id)
        return f"switches to {species_name}"


def _to_np(state):
    """Bulk convert state fields to numpy (cached per call)."""
    return {
        'active_idx': np.array(state.sides_active_idx),
        'species_id': np.array(state.sides_team_species_id),
        'hp': np.array(state.sides_team_hp),
        'max_hp': np.array(state.sides_team_max_hp),
        'status': np.array(state.sides_team_status),
        'move_ids': np.array(state.sides_team_move_ids),
        'fainted': np.array(state.sides_team_fainted),
        'types': np.array(state.sides_team_types),
        'boosts': np.array(state.sides_team_boosts),
        'ability_id': np.array(state.sides_team_ability_id),
        'item_id': np.array(state.sides_team_item_id),
        'pokemon_left': np.array(state.sides_pokemon_left),
        'side_conditions': np.array(state.sides_side_conditions),
    }


def print_team_summary(state, side, tables, label):
    """Print a team overview."""
    s = _to_np(state)
    active_idx = int(s['active_idx'][side])
    print(f"  {label}:")
    for slot in range(6):
        sid = int(s['species_id'][side, slot])
        if sid <= 0:
            continue
        name = get_species_name(tables, sid)
        hp = int(s['hp'][side, slot])
        max_hp = int(s['max_hp'][side, slot])
        types = [TYPE_NAMES[int(s['types'][side, slot, t])] for t in range(2) if int(s['types'][side, slot, t]) > 0]
        type_str = "/".join(types)
        ability = get_ability_name(tables, s['ability_id'][side, slot])
        item = get_item_name(tables, s['item_id'][side, slot])

        marker = " *" if slot == active_idx else "  "
        hp_bar = f"{hp}/{max_hp}"
        hp_pct = hp / max(max_hp, 1) * 100

        extras = []
        if ability:
            extras.append(ability)
        if item:
            extras.append(item)
        extra_str = f" ({', '.join(extras)})" if extras else ""

        moves = []
        for m in range(4):
            mid = int(s['move_ids'][side, slot, m])
            if mid > 0:
                moves.append(get_move_name(tables, mid))
        move_str = " | ".join(moves)

        fainted = bool(s['fainted'][side, slot])
        if fainted:
            status_str = " [FAINTED]"
        else:
            st = int(s['status'][side, slot])
            status_str = f" [{STATUS_NAMES[st]}]" if st > 0 else ""

        print(f"   {marker} {name:15s} {type_str:15s} HP: {hp_bar:>10s} ({hp_pct:5.1f}%){status_str}{extra_str}")
        print(f"        Moves: {move_str}")


def print_turn_state(state, tables):
    """Print compact turn state."""
    s = _to_np(state)
    for side, label in [(0, "BC Model"), (1, "Opponent")]:
        active_idx = int(s['active_idx'][side])
        sid = int(s['species_id'][side, active_idx])
        name = get_species_name(tables, sid)
        hp = int(s['hp'][side, active_idx])
        max_hp = int(s['max_hp'][side, active_idx])
        hp_pct = hp / max(max_hp, 1) * 100
        st = int(s['status'][side, active_idx])
        status_str = f" [{STATUS_NAMES[st]}]" if st > 0 else ""
        alive = int(s['pokemon_left'][side])
        boosts = s['boosts'][side, active_idx]
        boost_strs = []
        boost_names = ["Atk", "Def", "SpA", "SpD", "Spe", "Acc", "Eva"]
        for i, bn in enumerate(boost_names):
            b = int(boosts[i])
            if b > 0:
                boost_strs.append(f"+{b}{bn}")
            elif b < 0:
                boost_strs.append(f"{b}{bn}")
        boost_str = f" ({', '.join(boost_strs)})" if boost_strs else ""
        print(f"  {label:10s}: {name:15s} {hp:>3d}/{max_hp:<3d} ({hp_pct:5.1f}%){status_str}{boost_str}  [{alive} alive]")


def main():
    parser = argparse.ArgumentParser(description="Watch BC model play games")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/bc_final.pkl")
    parser.add_argument("--gen", type=int, default=4)
    parser.add_argument("--team-pool", type=str, default=None)
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vs", type=str, default="heuristic",
                        choices=["random", "heuristic"])
    parser.add_argument("--verbose", action="store_true",
                        help="Show full team info each turn")
    args = parser.parse_args()

    print(f"JAX backend: {jax.default_backend()}")

    from pokejax.env.pokejax_env import PokeJAXEnv
    from pokejax.rl.model import PokeTransformer
    from pokejax.rl.obs_builder import build_obs
    from pokejax.env.heuristic import smart_heuristic_action, random_action, _state_to_numpy
    from pokejax.env.action_mask import get_action_mask

    env = PokeJAXEnv(gen=args.gen, team_pool_path=args.team_pool)
    tables = env.tables
    model = PokeTransformer()

    print(f"Loading checkpoint: {args.checkpoint}")
    with open(args.checkpoint, "rb") as f:
        ckpt = pickle.load(f)
    params = ckpt["params"]

    # JIT compile
    print("JIT compiling (this takes a few minutes the first time)...")

    @jax.jit
    def get_model_action(params, int_ids, float_feats, legal_mask):
        log_probs, _, value = model.apply(
            params,
            int_ids[None],
            float_feats[None],
            legal_mask[None],
        )
        return jnp.argmax(log_probs[0]), log_probs[0], value[0]

    @jax.jit
    def jit_step(env_state, actions, step_key):
        return env.step(env_state, actions, step_key)

    @jax.jit
    def jit_obs(battle, reveal):
        return build_obs(battle, reveal, 0, tables)

    # Warm up JIT
    key = jax.random.PRNGKey(args.seed)
    key, warmup_key = jax.random.split(key)
    warmup_state, _ = env.reset(warmup_key)
    warmup_obs = jit_obs(warmup_state.battle, warmup_state.reveal)
    _ = get_model_action(params, warmup_obs["int_ids"], warmup_obs["float_feats"], warmup_obs["legal_mask"])
    warmup_actions = jnp.array([0, 0], dtype=jnp.int32)
    key, warmup_step_key = jax.random.split(key)
    _ = jit_step(warmup_state, warmup_actions, warmup_step_key)
    jax.block_until_ready(warmup_obs["int_ids"])
    print("JIT compiled. Starting games!\n")

    wins = 0
    for game in range(args.games):
        key, reset_key = jax.random.split(key)
        env_state, _ = env.reset(reset_key)
        state = env_state.battle
        reveal = env_state.reveal

        print("=" * 70)
        print(f"  GAME {game + 1}/{args.games}  —  BC Model (P1) vs {args.vs.title()} (P2)")
        print("=" * 70)

        # Show starting teams
        print_team_summary(state, 0, tables, "BC Model's Team")
        print()
        print_team_summary(state, 1, tables, "Opponent's Team")
        print()

        turn = 0
        while not bool(state.finished) and turn < 300:
            turn += 1
            print(f"--- Turn {turn} ---")
            print_turn_state(state, tables)

            # Model picks action
            obs = jit_obs(state, reveal)
            model_action_jax, log_probs, value = get_model_action(
                params, obs["int_ids"], obs["float_feats"], obs["legal_mask"]
            )
            model_action = int(model_action_jax)

            # Show model's reasoning
            lp = np.array(log_probs)
            mask = np.array(obs["legal_mask"])
            legal_actions = np.where(mask)[0]
            probs = np.exp(lp)

            # Opponent picks action
            np_cache = _state_to_numpy(state)
            if args.vs == "heuristic":
                opp_action = smart_heuristic_action(state, 1, tables, _np_cache=np_cache)
            else:
                opp_action = random_action(state, 1)

            model_action_str = format_action(model_action, state, 0, tables)
            opp_action_str = format_action(opp_action, state, 1, tables)

            # Show action probabilities for legal moves
            top_actions = sorted(legal_actions, key=lambda a: -probs[a])
            prob_strs = []
            for a in top_actions[:5]:
                a_str = format_action(a, state, 0, tables)
                p = probs[a] * 100
                marker = " <--" if a == model_action else ""
                prob_strs.append(f"    {a_str:30s} {p:5.1f}%{marker}")

            print(f"  BC Model  {model_action_str}")
            print(f"  Opponent  {opp_action_str}")
            print(f"  Model value: {float(value):.3f}")
            if args.verbose:
                print(f"  Action probabilities:")
                for ps in prob_strs:
                    print(ps)

            # Execute turn
            actions = jnp.array([model_action, opp_action], dtype=jnp.int32)
            key, step_key = jax.random.split(key)
            env_state, _, _, _, _ = jit_step(env_state, actions, step_key)

            prev_state = state
            state = env_state.battle
            reveal = env_state.reveal

            # Show what happened (HP changes, faints)
            s_prev = _to_np(prev_state)
            s_curr = _to_np(state)
            events = []
            for side, label in [(0, "BC"), (1, "Opp")]:
                for slot in range(6):
                    sid = int(s_curr['species_id'][side, slot])
                    if sid <= 0:
                        continue
                    name = get_species_name(tables, sid)
                    hp_before = int(s_prev['hp'][side, slot])
                    hp_after = int(s_curr['hp'][side, slot])
                    max_hp = int(s_curr['max_hp'][side, slot])
                    if hp_after != hp_before:
                        delta = hp_after - hp_before
                        pct = abs(delta) / max(max_hp, 1) * 100
                        if delta < 0:
                            events.append(f"  {label}'s {name} took {abs(delta)} damage ({pct:.0f}% HP)")
                        else:
                            events.append(f"  {label}'s {name} healed {delta} HP ({pct:.0f}%)")
                    # New status
                    st_before = int(s_prev['status'][side, slot])
                    st_after = int(s_curr['status'][side, slot])
                    if st_after != st_before and st_after > 0:
                        events.append(f"  {label}'s {name} is now {STATUS_NAMES[st_after]}!")
                    # Fainted
                    if not bool(s_prev['fainted'][side, slot]) and bool(s_curr['fainted'][side, slot]):
                        events.append(f"  {label}'s {name} fainted!")

            if events:
                for e in events:
                    print(e)
            print()

        # Game result
        if bool(state.finished):
            winner = int(state.winner)
            if winner == 0:
                wins += 1
                result = "BC MODEL WINS!"
            elif winner == 1:
                result = "OPPONENT WINS!"
            else:
                result = "DRAW!"
        else:
            result = "TIMEOUT (300 turns)"

        s_final = _to_np(state)
        print("=" * 70)
        print(f"  RESULT: {result}  (Turn {turn})")
        print(f"  BC remaining: {int(s_final['pokemon_left'][0])}/6")
        print(f"  Opp remaining: {int(s_final['pokemon_left'][1])}/6")
        print("=" * 70)
        print()

    print(f"\n{'=' * 70}")
    print(f"  FINAL SCORE: {wins}/{args.games} wins ({wins/args.games:.0%})")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
