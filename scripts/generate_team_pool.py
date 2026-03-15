"""
Generate a pre-computed pool of Gen 4 random battle teams.

Uses the engine's own Tables for species/move/ability/item ID mappings,
ensuring the team pool is compatible with the battle engine.

Reads gen4randombattle.json from PokemonShowdownClaude/data for the
set/role/movepool definitions, but maps all names to engine IDs.

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
import re
import sys
from pathlib import Path

import numpy as np

# Fields per Pokemon in the pool array
# species_id, ability_id, item_id, type1, type2,
# base_hp, base_atk, base_def, base_spa, base_spd, base_spe,
# max_hp, move_id_0..3, move_pp_0..3, level, weight_hg
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


def normalize_name(name: str) -> str:
    """Normalize a Pokemon/move/ability/item name for lookup.
    Converts to lowercase, removes non-alphanumeric chars.
    """
    return re.sub(r'[^a-z0-9]', '', name.lower())


def compute_stat(base: int, iv: int, ev: int, level: int, nature_mult: float = 1.0) -> int:
    return int((int((2 * base + iv + ev // 4) * level / 100) + 5) * nature_mult)


def compute_hp(base: int, iv: int, ev: int, level: int) -> int:
    if base == 1:  # Shedinja
        return 1
    return int((2 * base + iv + ev // 4) * level / 100) + level + 10


def build_normalized_lookup(name_to_id: dict) -> dict:
    """Pre-build a normalized name → id lookup for fast matching."""
    lookup = {}
    for k, v in name_to_id.items():
        lookup[k] = v
        lookup[normalize_name(k)] = v
    return lookup


def find_id(name: str, lookup: dict) -> int:
    """Find ID using exact match, then normalized match."""
    if name in lookup:
        return lookup[name]
    norm = normalize_name(name)
    if norm in lookup:
        return lookup[norm]
    return 0  # Not found


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

    # Load the engine's tables for ID mapping
    print("Loading engine tables...")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pokejax.data.tables import load_tables
    tables = load_tables(4)
    print(f"  Species: {len(tables.species_name_to_id)}, "
          f"Moves: {len(tables.move_name_to_id)}, "
          f"Abilities: {len(tables.ability_name_to_id)}, "
          f"Items: {len(tables.item_name_to_id)}")

    # Pre-build normalized lookups for fast matching
    species_lookup = build_normalized_lookup(tables.species_name_to_id)
    move_lookup = build_normalized_lookup(tables.move_name_to_id)
    ability_lookup = build_normalized_lookup(tables.ability_name_to_id)
    item_lookup = build_normalized_lookup(tables.item_name_to_id)

    # Convert JAX device arrays to numpy for fast CPU access
    moves_np = np.array(tables.moves)

    # Load randbats and base stats
    print(f"Loading randbats data from {args.bot_data_dir}...")
    with open(os.path.join(args.bot_data_dir, 'gen4randombattle.json')) as f:
        randbats = json.load(f)

    base_stats_path = os.path.join(args.bot_data_dir, 'gen4_base_stats.json')
    base_stats = {}
    if os.path.exists(base_stats_path):
        with open(base_stats_path) as f:
            base_stats = json.load(f)

    print(f"  {len(randbats)} species in gen4randombattle.json")

    # Validate: check how many species/moves map successfully
    species_mapped = 0
    species_unmapped = []
    for name in randbats:
        sid = find_id(name, species_lookup)
        if sid > 0:
            species_mapped += 1
        else:
            species_unmapped.append(name)

    print(f"  Species mapped: {species_mapped}/{len(randbats)}")
    if species_unmapped and len(species_unmapped) <= 20:
        print(f"  Unmapped: {species_unmapped}")

    # Generate teams
    print(f"Generating {args.n_teams} teams...")
    species_list = list(randbats.keys())
    pool = np.zeros((args.n_teams, 6, FIELDS_PER_MON), dtype=np.int16)
    skipped = 0

    for i in range(args.n_teams):
        # Sample 6 unique species
        chosen = random.sample(species_list, min(6, len(species_list)))

        for slot, species_name in enumerate(chosen):
            data = randbats[species_name]
            level = data.get('level', 80)

            # Species ID from engine tables
            species_id = find_id(species_name, species_lookup)

            # Pick random role
            roles = data.get('roles', {})
            if not roles:
                continue
            role_name = random.choice(list(roles.keys()))
            role = roles[role_name]

            # Ability
            abilities = role.get('abilities', data.get('abilities', ['']))
            ability_name = random.choice(abilities) if abilities else ''
            ability_id = find_id(ability_name, ability_lookup)

            # Item
            items = role.get('items', data.get('items', ['']))
            item_name = random.choice(items) if items else ''
            item_id = find_id(item_name, item_lookup)

            # Moves (pick 4 from role's move pool)
            move_pool = role.get('moves', [])
            if len(move_pool) > 4:
                chosen_moves = random.sample(move_pool, 4)
            else:
                chosen_moves = move_pool[:4]
            while len(chosen_moves) < 4:
                chosen_moves.append('')

            move_ids = []
            move_pps = []
            for m in chosen_moves:
                mid = find_id(m, move_lookup)
                move_ids.append(mid)
                # Get PP from engine's move data
                if mid > 0 and mid < len(moves_np):
                    pp = int(moves_np[mid, 5])  # MF_PP = 5
                    # Apply PP ups (randbats get max PP ups = pp * 8/5)
                    pp = pp * 8 // 5
                    move_pps.append(min(pp, 64))
                else:
                    move_pps.append(24)  # default

            # EVs/IVs
            evs = role.get('evs', data.get('evs', {}))
            ivs = role.get('ivs', data.get('ivs', {}))
            ev_hp  = evs.get('hp', 85)
            iv_hp  = ivs.get('hp', 31)

            # Get base stats from base_stats.json
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

            # Compute max HP
            max_hp = compute_hp(b_hp, iv_hp, ev_hp, level)

            # Weight
            weight_hg = int(bs.get('weightkg', 80) * 10)

            pool[i, slot] = [
                species_id, ability_id, item_id, type1, type2,
                b_hp, b_atk, b_def, b_spa, b_spd, b_spe,
                max_hp,
                *move_ids,
                *move_pps,
                level,
                weight_hg,
            ]

        if (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{args.n_teams}")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(args.output, teams=pool, field_names=FIELD_NAMES)
    print(f"Saved to {args.output} ({os.path.getsize(args.output) / 1024 / 1024:.1f} MB)")


if __name__ == '__main__':
    main()
