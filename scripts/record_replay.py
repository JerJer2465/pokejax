#!/usr/bin/env python3
"""Record pokejax battle replays for debugging.

Runs battles and saves turn-by-turn state snapshots to replay files
that can be loaded in scripts/replay_viewer.py.

Run in WSL (requires JAX/CUDA):
    # Heuristic vs heuristic (no checkpoint needed):
    wsl bash -c "cd /mnt/c/Users/jerry/Documents/Coding/pokejax && python3 scripts/record_replay.py --games 5"

    # Model vs heuristic:
    wsl bash -c "cd /mnt/c/Users/jerry/Documents/Coding/pokejax && python3 scripts/record_replay.py --checkpoint checkpoints/ppo_latest.pkl --games 5"

    # Both players are the model:
    wsl bash -c "cd /mnt/c/Users/jerry/Documents/Coding/pokejax && python3 scripts/record_replay.py --checkpoint checkpoints/ppo_latest.pkl --vs self"
"""

import argparse
import os
import sys
import pickle
import datetime

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Constants (mirrored from types.py for standalone use)
# ---------------------------------------------------------------------------
STATUS_NAMES = {
    0: '', 1: 'burned', 2: 'poisoned', 3: 'badly poisoned',
    4: 'fell asleep', 5: 'frozen solid', 6: 'paralyzed',
}
WEATHER_NAMES = {0: '', 1: 'Harsh Sunlight', 2: 'Rain', 3: 'Sandstorm', 4: 'Hail'}
TERRAIN_NAMES = {0: '', 1: 'Electric Terrain', 2: 'Grassy Terrain',
                 3: 'Misty Terrain', 4: 'Psychic Terrain'}
SC_NAMES = [
    'Spikes', 'Toxic Spikes', 'Stealth Rock', 'Sticky Web',
    'Reflect', 'Light Screen', 'Aurora Veil', 'Tailwind',
    'Safeguard', 'Mist',
]
BOOST_NAMES = ['Atk', 'Def', 'SpA', 'SpD', 'Spe', 'Acc', 'Eva']
VOL_NAMES = {
    0:  ('Confused',  True),
    1:  ('Flinch',    False),
    2:  ('Trapped',   True),
    3:  ('Seeded',    False),
    4:  ('Sub',       True),
    5:  ('Protect',   False),
    6:  ('Encore',    True),
    7:  ('Taunt',     True),
    8:  ('Torment',   False),
    9:  ('Disable',   True),
    10: ('Endure',    False),
    11: ('MagicCoat', False),
    12: ('Snatch',    False),
    13: ('Ingrain',   False),
    15: ('HealBlock', True),
    16: ('Embargo',   True),
    17: ('Charging',  False),
    18: ('Recharge',  False),
    19: ('Locked',    True),
    20: ('ChoiceLock',True),
    21: ('FocusNrg',  False),
    22: ('Minimize',  False),
    23: ('Curse',     False),
    24: ('Nightmare', False),
    25: ('Infatuat',  False),
    26: ('Yawn',      True),
    27: ('DestinyBnd',False),
    28: ('Grudge',    False),
    30: ('Perish',    True),
}


# ---------------------------------------------------------------------------
# State snapshot
# ---------------------------------------------------------------------------

def state_to_snapshot(state) -> dict:
    """Convert BattleState to a pure-numpy dict for serialization."""
    return {
        'species_id':       np.array(state.sides_team_species_id),
        'ability_id':       np.array(state.sides_team_ability_id),
        'item_id':          np.array(state.sides_team_item_id),
        'types':            np.array(state.sides_team_types),
        'base_stats':       np.array(state.sides_team_base_stats),
        'hp':               np.array(state.sides_team_hp),
        'max_hp':           np.array(state.sides_team_max_hp),
        'boosts':           np.array(state.sides_team_boosts),
        'move_ids':         np.array(state.sides_team_move_ids),
        'move_pp':          np.array(state.sides_team_move_pp),
        'move_max_pp':      np.array(state.sides_team_move_max_pp),
        'move_disabled':    np.array(state.sides_team_move_disabled),
        'status':           np.array(state.sides_team_status),
        'status_turns':     np.array(state.sides_team_status_turns),
        'sleep_turns':      np.array(state.sides_team_sleep_turns),
        'volatiles':        np.array(state.sides_team_volatiles),
        'volatile_data':    np.array(state.sides_team_volatile_data),
        'active_idx':       np.array(state.sides_active_idx),
        'fainted':          np.array(state.sides_team_fainted),
        'pokemon_left':     np.array(state.sides_pokemon_left),
        'side_conditions':  np.array(state.sides_side_conditions),
        'level':            np.array(state.sides_team_level),
        'weather':          int(state.field.weather),
        'weather_turns':    int(state.field.weather_turns),
        'terrain':          int(state.field.terrain),
        'terrain_turns':    int(state.field.terrain_turns),
        'trick_room':       int(state.field.trick_room),
        'gravity':          int(state.field.gravity),
        'turn':             int(state.turn),
        'finished':         bool(state.finished),
        'winner':           int(state.winner),
    }


# ---------------------------------------------------------------------------
# Event computation (diff two snapshots)
# ---------------------------------------------------------------------------

def _species_name(snap, side, slot, tables):
    sid = int(snap['species_id'][side, slot])
    if 0 < sid < len(tables['species_names']):
        return tables['species_names'][sid]
    return f'???({sid})'


def _move_name(snap, side, slot, move_slot, tables):
    mid = int(snap['move_ids'][side, slot, move_slot])
    if 0 < mid < len(tables['move_names']):
        return tables['move_names'][mid]
    return f'???({mid})'


def compute_events(s_before: dict, s_after: dict, actions: list,
                   tables: dict, player_names: list) -> list:
    """Diff two state snapshots to produce (tag, text) event tuples."""
    events = []
    SIDE = player_names  # e.g. ['P1', 'P2']

    # 1. Actions declared this turn
    for side in [0, 1]:
        a = int(actions[side])
        active = int(s_before['active_idx'][side])
        pname = _species_name(s_before, side, active, tables)
        if a < 4:
            mname = _move_name(s_before, side, active, a, tables)
            events.append(('action', f"{SIDE[side]}'s {pname} used {mname}"))
        else:
            slot = a - 4
            swname = _species_name(s_before, side, slot, tables)
            events.append(('action', f"{SIDE[side]} switched in {swname}!"))

    # 2. HP changes
    for side in [0, 1]:
        for slot in range(6):
            sid = int(s_before['species_id'][side, slot])
            if sid <= 0:
                continue
            name = _species_name(s_before, side, slot, tables)
            hp_before = int(s_before['hp'][side, slot])
            hp_after  = int(s_after['hp'][side, slot])
            max_hp    = max(int(s_after['max_hp'][side, slot]), 1)
            delta = hp_after - hp_before
            if delta < 0:
                pct = abs(delta) / max_hp * 100
                events.append(('damage',
                    f"{SIDE[side]}'s {name} took {abs(delta)} damage ({pct:.0f}%)"))
            elif delta > 0:
                pct = delta / max_hp * 100
                events.append(('heal',
                    f"{SIDE[side]}'s {name} restored {delta} HP ({pct:.0f}%)"))

    # 3. Faints
    for side in [0, 1]:
        for slot in range(6):
            sid = int(s_before['species_id'][side, slot])
            if sid <= 0:
                continue
            name = _species_name(s_before, side, slot, tables)
            if not bool(s_before['fainted'][side, slot]) and bool(s_after['fainted'][side, slot]):
                events.append(('faint', f"{SIDE[side]}'s {name} fainted!"))

    # 4. Status changes
    for side in [0, 1]:
        for slot in range(6):
            sid = int(s_before['species_id'][side, slot])
            if sid <= 0:
                continue
            name = _species_name(s_before, side, slot, tables)
            st_before = int(s_before['status'][side, slot])
            st_after  = int(s_after['status'][side, slot])
            if st_after != st_before:
                if st_after > 0:
                    events.append(('status',
                        f"{SIDE[side]}'s {name} was {STATUS_NAMES[st_after]}!"))
                elif st_before > 0:
                    events.append(('status',
                        f"{SIDE[side]}'s {name} was cured of its status!"))

    # 5. Boost changes
    for side in [0, 1]:
        for slot in range(6):
            sid = int(s_before['species_id'][side, slot])
            if sid <= 0:
                continue
            name = _species_name(s_before, side, slot, tables)
            for bi, bname in enumerate(BOOST_NAMES):
                b_before = int(s_before['boosts'][side, slot, bi])
                b_after  = int(s_after['boosts'][side, slot, bi])
                d = b_after - b_before
                if d >= 2:
                    events.append(('boost',
                        f"{SIDE[side]}'s {name}'s {bname} sharply rose!"))
                elif d == 1:
                    events.append(('boost',
                        f"{SIDE[side]}'s {name}'s {bname} rose!"))
                elif d == -1:
                    events.append(('boost',
                        f"{SIDE[side]}'s {name}'s {bname} fell!"))
                elif d <= -2:
                    events.append(('boost',
                        f"{SIDE[side]}'s {name}'s {bname} harshly fell!"))

    # 6. Weather changes
    w_before = int(s_before['weather'])
    w_after  = int(s_after['weather'])
    if w_before != w_after:
        if w_after == 0:
            events.append(('field',
                f"The {WEATHER_NAMES.get(w_before, 'weather').lower()} subsided!"))
        else:
            events.append(('field', f"{WEATHER_NAMES[w_after]} started!"))

    # 7. Terrain changes
    t_before = int(s_before['terrain'])
    t_after  = int(s_after['terrain'])
    if t_before != t_after:
        if t_after == 0:
            events.append(('field', "The terrain returned to normal!"))
        else:
            events.append(('field', f"{TERRAIN_NAMES[t_after]} was established!"))

    # 8. Trick Room
    tr_before = int(s_before['trick_room'])
    tr_after  = int(s_after['trick_room'])
    if tr_before == 0 and tr_after > 0:
        events.append(('field', "Trick Room was set up!"))
    elif tr_before > 0 and tr_after == 0:
        events.append(('field', "Trick Room ended!"))

    # 9. Side conditions
    for side in [0, 1]:
        for ci, cname in enumerate(SC_NAMES):
            c_before = int(s_before['side_conditions'][side, ci])
            c_after  = int(s_after['side_conditions'][side, ci])
            if c_after > c_before:
                events.append(('field',
                    f"{cname} set on {SIDE[side]}'s side!"))
            elif c_after == 0 and c_before > 0:
                events.append(('field',
                    f"{cname} on {SIDE[side]}'s side ended!"))

    return events


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class HeuristicAgent:
    def __init__(self, tables):
        from pokejax.env.heuristic import smart_heuristic_action, _state_to_numpy
        self._act = smart_heuristic_action
        self._cache = _state_to_numpy
        self.tables = tables

    def choose_action(self, state, side, np_cache=None):
        if np_cache is None:
            np_cache = self._cache(state)
        return int(self._act(state, side, self.tables, _np_cache=np_cache))


class RandomAgent:
    def choose_action(self, state, side, np_cache=None):
        from pokejax.env.heuristic import random_action
        return int(random_action(state, side))


class ModelAgent:
    def __init__(self, checkpoint_path, gen, tables):
        import pickle as pkl
        from pokejax.rl.model import create_model
        from pokejax.rl.obs_builder import build_obs

        with open(checkpoint_path, 'rb') as f:
            ckpt = pkl.load(f)
        arch = ckpt.get('arch', 'transformer')
        self.model = create_model(arch)
        self.params = ckpt['params']
        self.tables = tables
        self._build_obs = build_obs

        @jax.jit
        def _forward(params, int_ids, float_feats, legal_mask):
            log_probs, _, _ = self.model.apply(
                params, int_ids[None], float_feats[None], legal_mask[None]
            )
            return jnp.argmax(log_probs[0])

        self._forward = _forward

    def choose_action(self, state, side, np_cache=None):
        from pokejax.rl.obs_builder import build_obs
        obs = build_obs(state, None, side, self.tables)
        action = self._forward(self.params, obs['int_ids'], obs['float_feats'], obs['legal_mask'])
        return int(action)


# ---------------------------------------------------------------------------
# Core recording loop
# ---------------------------------------------------------------------------

def record_one_game(env, tables_data, tables_obj, key, agent0, agent1,
                    game_idx, player_names, jit_step):
    """Run one complete battle and return replay dict."""
    key, reset_key = jax.random.split(key)
    env_state, _ = env.reset(reset_key)
    state = env_state.battle

    turns = []
    max_turns = 300

    while not bool(state.finished) and len(turns) < max_turns:
        snap_before = state_to_snapshot(state)

        # Get numpy cache for heuristic (shared between agents if both heuristic)
        try:
            from pokejax.env.heuristic import _state_to_numpy
            np_cache = _state_to_numpy(state)
        except Exception:
            np_cache = None

        a0 = agent0.choose_action(state, 0, np_cache)
        a1 = agent1.choose_action(state, 1, np_cache)

        actions_jax = jnp.array([a0, a1], dtype=jnp.int32)
        key, step_key = jax.random.split(key)
        env_state, _, _, _, _ = jit_step(env_state, actions_jax, step_key)
        state = env_state.battle

        snap_after = state_to_snapshot(state)
        events = compute_events(snap_before, snap_after, [a0, a1],
                                tables_data, player_names)

        turns.append({
            'state':       snap_before,
            'actions':     [a0, a1],
            'events':      events,
            'state_after': snap_after,
        })

    # Determine result
    winner = int(state.winner)
    if not bool(state.finished):
        result = 'Timeout (300 turns)'
    elif winner == 0:
        result = f'{player_names[0]} wins'
    elif winner == 1:
        result = f'{player_names[1]} wins'
    else:
        result = 'Draw'

    return {
        'turns':       turns,
        'result':      result,
        'total_turns': len(turns),
        'tables':      tables_data,
        'metadata': {
            'p0_name': player_names[0],
            'p1_name': player_names[1],
            'gen':     env.cfg.gen,
            'date':    datetime.datetime.now().isoformat(),
            'game_idx': game_idx,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Record pokejax battle replays')
    parser.add_argument('--games', type=int, default=5,
                        help='Number of games to record')
    parser.add_argument('--gen', type=int, default=4,
                        help='Pokemon generation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint for P1 (default: heuristic)')
    parser.add_argument('--vs', type=str, default='heuristic',
                        choices=['heuristic', 'random', 'self'],
                        help='P2 opponent type')
    parser.add_argument('--output', type=str, default='replays/',
                        help='Output directory for replay files')
    parser.add_argument('--team-pool', type=str, default=None)
    args = parser.parse_args()

    print(f'JAX backend: {jax.default_backend()}')

    from pokejax.env.pokejax_env import PokeJAXEnv

    env = PokeJAXEnv(gen=args.gen, team_pool_path=args.team_pool)
    tables_obj = env.tables

    # Build tables dict for serialization (plain Python, no JAX)
    tables_data = {
        'species_names': list(tables_obj.species_names),
        'move_names':    list(tables_obj.move_names),
        'ability_names': list(tables_obj.ability_names),
        'item_names':    list(tables_obj.item_names),
    }

    # Build agents
    if args.checkpoint:
        print(f'Loading checkpoint: {args.checkpoint}')
        agent0 = ModelAgent(args.checkpoint, args.gen, tables_obj)
        p0_name = 'Model'
    else:
        agent0 = HeuristicAgent(tables_obj)
        p0_name = 'Heuristic'

    if args.vs == 'self' and args.checkpoint:
        agent1 = ModelAgent(args.checkpoint, args.gen, tables_obj)
        p1_name = 'Model'
    elif args.vs == 'random':
        agent1 = RandomAgent()
        p1_name = 'Random'
    else:
        agent1 = HeuristicAgent(tables_obj)
        p1_name = 'Heuristic'

    player_names = [p0_name, p1_name]
    print(f'Recording {args.games} games: {p0_name} vs {p1_name}')

    # JIT compile step
    print('JIT compiling...')

    @jax.jit
    def jit_step(env_state, actions, key):
        return env.step(env_state, actions, key)

    # Warm up
    key = jax.random.PRNGKey(args.seed)
    key, warmup_key = jax.random.split(key)
    warmup_state, _ = env.reset(warmup_key)
    warmup_actions = jnp.array([0, 0], dtype=jnp.int32)
    key, warmup_step_key = jax.random.split(key)
    _ = jit_step(warmup_state, warmup_actions, warmup_step_key)
    jax.block_until_ready(warmup_state.battle.turn)
    print('JIT compiled.\n')

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Record games
    for game_idx in range(args.games):
        key, game_key = jax.random.split(key)
        print(f'Recording game {game_idx + 1}/{args.games}...')

        replay = record_one_game(
            env, tables_data, tables_obj, game_key,
            agent0, agent1, game_idx, player_names, jit_step,
        )

        # Save replay
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'game_{game_idx:04d}_{timestamp}.pkl'
        output_path = os.path.join(args.output, filename)
        with open(output_path, 'wb') as f:
            pickle.dump(replay, f)

        print(f'  Result: {replay["result"]}  ({replay["total_turns"]} turns)')
        print(f'  Saved: {output_path}')

    print(f'\nDone. {args.games} replays saved to {args.output}')


if __name__ == '__main__':
    main()
