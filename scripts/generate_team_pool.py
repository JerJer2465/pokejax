"""
Generate a pre-computed pool of Gen 4 random battle teams.

Reads gen4randombattle.json and gen4_base_stats.json from the
PokemonShowdownClaude data directory, then generates N_TEAMS teams
using the same sampling logic as Showdown's random team builder.

Output: data/team_pool.npz containing:
  - teams: int16[N_TEAMS, 6, FIELDS_PER_MON]
  - field_names: list of field names

Usage:
    python scripts/generate_team_pool.py --n-teams 100000
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

# Fields per Pokemon in the pool array
# species_id, ability_id, item_id, type1, type2,
# base_hp, base_atk, base_def, base_spa, base_spd, base_spe,
# max_hp, move_id_0..3, move_pp_0..3, level
FIELDS_PER_MON = 22

FIELD_NAMES = [
    'species_id', 'ability_id', 'item_id', 'type1', 'type2',
    'base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe',
    'max_hp',
    'move_id_0', 'move_id_1', 'move_id_2', 'move_id_3',
    'move_pp_0', 'move_pp_1', 'move_pp_2', 'move_pp_3',
    'level', 'weight_hg',
]

# Type name → index (matches types.py)
TYPE_MAP = {
    '???': 0, 'Normal': 1, 'Fire': 2, 'Water': 3, 'Electric': 4,
    'Grass': 5, 'Ice': 6, 'Fighting': 7, 'Poison': 8, 'Ground': 9,
    'Flying': 10, 'Psychic': 11, 'Bug': 12, 'Rock': 13, 'Ghost': 14,
    'Dragon': 15, 'Dark': 16, 'Steel': 17, 'Fairy': 18,
}


def load_data(bot_data_dir: str):
    """Load randbats JSON and base stats."""
    with open(os.path.join(bot_data_dir, 'gen4randombattle.json')) as f:
        randbats = json.load(f)

    # Try to load base stats if available
    base_stats_path = os.path.join(bot_data_dir, 'gen4_base_stats.json')
    base_stats = {}
    if os.path.exists(base_stats_path):
        with open(base_stats_path) as f:
            base_stats = json.load(f)

    return randbats, base_stats


def compute_stat(base: int, iv: int, ev: int, level: int, nature_mult: float = 1.0) -> int:
    """Compute a non-HP stat."""
    return int((int((2 * base + iv + ev // 4) * level / 100) + 5) * nature_mult)


def compute_hp(base: int, iv: int, ev: int, level: int) -> int:
    """Compute HP stat."""
    if base == 1:  # Shedinja
        return 1
    return int((2 * base + iv + ev // 4) * level / 100) + level + 10


def build_name_to_id_maps(randbats: dict, base_stats: dict):
    """Build name→ID mappings for species, abilities, items, moves."""
    species_names = sorted(randbats.keys())
    species_to_id = {name: i for i, name in enumerate(species_names)}

    # Collect all abilities, items, moves
    all_abilities = set()
    all_items = set()
    all_moves = set()

    for species, data in randbats.items():
        all_abilities.update(data.get('abilities', []))
        all_items.update(data.get('items', []))
        for role_data in data.get('roles', {}).values():
            all_abilities.update(role_data.get('abilities', []))
            all_items.update(role_data.get('items', []))
            all_moves.update(role_data.get('moves', []))

    ability_to_id = {name: i + 1 for i, name in enumerate(sorted(all_abilities))}
    item_to_id = {name: i + 1 for i, name in enumerate(sorted(all_items))}
    move_to_id = {name: i + 1 for i, name in enumerate(sorted(all_moves))}

    return species_to_id, ability_to_id, item_to_id, move_to_id


def sample_team(randbats: dict, base_stats: dict,
                species_to_id: dict, ability_to_id: dict,
                item_to_id: dict, move_to_id: dict) -> np.ndarray:
    """Sample one random team. Returns int16[6, FIELDS_PER_MON]."""
    species_list = list(randbats.keys())
    team = np.zeros((6, FIELDS_PER_MON), dtype=np.int16)

    # Sample 6 unique species
    chosen = random.sample(species_list, min(6, len(species_list)))

    for slot, species_name in enumerate(chosen):
        data = randbats[species_name]
        level = data.get('level', 80)

        # Pick random role
        roles = data.get('roles', {})
        if not roles:
            continue
        role_name = random.choice(list(roles.keys()))
        role = roles[role_name]

        # Ability
        abilities = role.get('abilities', data.get('abilities', ['']))
        ability_name = random.choice(abilities) if abilities else ''
        ability_id = ability_to_id.get(ability_name, 0)

        # Item
        items = role.get('items', data.get('items', ['']))
        item_name = random.choice(items) if items else ''
        item_id = item_to_id.get(item_name, 0)

        # Moves (pick 4 from role's move pool)
        move_pool = role.get('moves', [])
        if len(move_pool) > 4:
            chosen_moves = random.sample(move_pool, 4)
        else:
            chosen_moves = move_pool[:4]
        # Pad to 4
        while len(chosen_moves) < 4:
            chosen_moves.append('')

        move_ids = [move_to_id.get(m, 0) for m in chosen_moves]

        # EVs/IVs
        evs = role.get('evs', data.get('evs', {}))
        ivs = role.get('ivs', data.get('ivs', {}))
        ev_hp  = evs.get('hp', 85)
        ev_atk = evs.get('atk', 85)
        ev_def = evs.get('def', 85)
        ev_spa = evs.get('spa', 85)
        ev_spd = evs.get('spd', 85)
        ev_spe = evs.get('spe', 85)
        iv_hp  = ivs.get('hp', 31)
        iv_atk = ivs.get('atk', 31)
        iv_def = ivs.get('def', 31)
        iv_spa = ivs.get('spa', 31)
        iv_spd = ivs.get('spd', 31)
        iv_spe = ivs.get('spe', 31)

        # Get base stats
        bs = base_stats.get(species_name, {})
        b_hp  = bs.get('hp', 80)
        b_atk = bs.get('atk', 80)
        b_def = bs.get('def', 80)
        b_spa = bs.get('spa', 80)
        b_spd = bs.get('spd', 80)
        b_spe = bs.get('spe', 80)

        # Types
        types = bs.get('types', ['Normal'])
        type1 = TYPE_MAP.get(types[0], 1)
        type2 = TYPE_MAP.get(types[1], 0) if len(types) > 1 else 0

        # Compute stats
        max_hp = compute_hp(b_hp, iv_hp, ev_hp, level)
        # Compute battle stats (stored as base_stats in the state)
        stat_atk = compute_stat(b_atk, iv_atk, ev_atk, level)
        stat_def = compute_stat(b_def, iv_def, ev_def, level)
        stat_spa = compute_stat(b_spa, iv_spa, ev_spa, level)
        stat_spd = compute_stat(b_spd, iv_spd, ev_spd, level)
        stat_spe = compute_stat(b_spe, iv_spe, ev_spe, level)

        # Move PP (simplified: all moves have 16 PP with 3/5 PP ups = ~24)
        move_pp = [24, 24, 24, 24]

        # Weight
        weight_hg = bs.get('weightkg', 80) * 10  # convert kg to hectograms

        # Fill team array
        team[slot] = [
            species_to_id.get(species_name, 0),
            ability_id, item_id, type1, type2,
            b_hp, b_atk, b_def, b_spa, b_spd, b_spe,  # raw base stats
            max_hp,
            *move_ids,
            *move_pp,
            level,
            int(weight_hg),
        ]

    return team


def main():
    parser = argparse.ArgumentParser(description='Generate Gen 4 random battle team pool')
    parser.add_argument('--n-teams', type=int, default=100000,
                        help='Number of teams to generate')
    parser.add_argument('--bot-data-dir', type=str,
                        default=str(Path(__file__).resolve().parent.parent.parent /
                                    'PokemonShowdownClaude' / 'data'),
                        help='Path to PokemonShowdownClaude/data directory')
    parser.add_argument('--output', type=str,
                        default=str(Path(__file__).resolve().parent.parent / 'data' / 'team_pool.npz'),
                        help='Output path')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading data from {args.bot_data_dir}...")
    randbats, base_stats = load_data(args.bot_data_dir)
    print(f"  {len(randbats)} species in gen4randombattle.json")

    species_to_id, ability_to_id, item_to_id, move_to_id = \
        build_name_to_id_maps(randbats, base_stats)

    print(f"  {len(ability_to_id)} abilities, {len(item_to_id)} items, {len(move_to_id)} moves")

    print(f"Generating {args.n_teams} teams...")
    pool = np.zeros((args.n_teams, 6, FIELDS_PER_MON), dtype=np.int16)
    for i in range(args.n_teams):
        pool[i] = sample_team(randbats, base_stats,
                              species_to_id, ability_to_id,
                              item_to_id, move_to_id)
        if (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{args.n_teams}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(
        args.output,
        teams=pool,
        field_names=FIELD_NAMES,
        species_names=sorted(randbats.keys()),
        ability_names=[''] + sorted(ability_to_id.keys()),
        item_names=[''] + sorted(item_to_id.keys()),
        move_names=[''] + sorted(move_to_id.keys()),
    )
    print(f"Saved to {args.output} ({os.path.getsize(args.output) / 1024 / 1024:.1f} MB)")


if __name__ == '__main__':
    main()
