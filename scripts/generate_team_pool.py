"""
Generate a pre-computed pool of Gen 4 random battle teams.

Faithfully ports Pokemon Showdown's Gen 4 random team generation logic:
  - Move selection: STAB enforcement, move pairing, incompatible moves,
    role-based enforcement (setup, recovery, hazards, coverage)
  - Item selection: getPriorityItem + getItem logic
  - Ability selection: getAbility + shouldCullAbility logic
  - Team-level constraints: type balance, weakness limits, teamDetails
    tracking (Stealth Rock, Rapid Spin, screens, weather, etc.)
  - EV/IV optimization: SR-damage HP tuning, 0 Atk for special attackers
  - Species compatibility (no Shedinja+Tyranitar, etc.)

Uses sets.json (the actual Showdown data source) for movepool/role definitions,
and the engine's own Tables for species/move/ability/item ID mappings.

Output: data/team_pool.npz containing:
  - teams: int16[N_TEAMS, 6, FIELDS_PER_MON]
  - field_names: list of field names

This runs ONCE on CPU as a preprocessing step. At runtime, team_gen.py
samples from this pool using fully vectorized JAX PRNG on GPU.

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
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Output array layout
# ---------------------------------------------------------------------------

FIELDS_PER_MON = 22

FIELD_NAMES = [
    'species_id', 'ability_id', 'item_id', 'type1', 'type2',
    'base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe',
    'max_hp',
    'move_id_0', 'move_id_1', 'move_id_2', 'move_id_3',
    'move_pp_0', 'move_pp_1', 'move_pp_2', 'move_pp_3',
    'level', 'weight_hg',
]

TYPE_MAP = {
    '???': 0, 'Normal': 1, 'Fire': 2, 'Water': 3, 'Electric': 4,
    'Grass': 5, 'Ice': 6, 'Fighting': 7, 'Poison': 8, 'Ground': 9,
    'Flying': 10, 'Psychic': 11, 'Bug': 12, 'Rock': 13, 'Ghost': 14,
    'Dragon': 15, 'Dark': 16, 'Steel': 17, 'Fairy': 18,
}
TYPE_MAP_REV = {v: k for k, v in TYPE_MAP.items()}

# Move data field indices (must match extractor.py / damage.py)
MF_BASE_POWER = 0
MF_ACCURACY = 1
MF_TYPE = 2
MF_CATEGORY = 3  # 0=Physical, 1=Special, 2=Status
MF_PRIORITY = 4
MF_PP = 5
MF_RECOIL_NUM = 16
MF_RECOIL_DEN = 17
MF_MULTIHIT_MIN = 18
MF_MULTIHIT_MAX = 19

# ---------------------------------------------------------------------------
# Showdown constants (ported from gen4/teams.ts)
# ---------------------------------------------------------------------------

RECOVERY_MOVES = {
    'healorder', 'milkdrink', 'moonlight', 'morningsun', 'recover',
    'roost', 'slackoff', 'softboiled', 'synthesis',
}

PHYSICAL_SETUP = {
    'bellydrum', 'bulkup', 'curse', 'dragondance', 'howl', 'meditate',
    'screech', 'swordsdance',
}

SETUP = {
    'acidarmor', 'agility', 'bellydrum', 'bulkup', 'calmmind', 'curse',
    'dragondance', 'growth', 'howl', 'irondefense', 'meditate', 'nastyplot',
    'raindance', 'rockpolish', 'sunnyday', 'swordsdance', 'tailglow',
}

NO_STAB = {
    'aquajet', 'bulletpunch', 'chatter', 'eruption', 'explosion', 'fakeout',
    'focuspunch', 'futuresight', 'iceshard', 'icywind', 'knockoff',
    'machpunch', 'pluck', 'pursuit', 'quickattack', 'rapidspin', 'reversal',
    'selfdestruct', 'shadowsneak', 'skyattack', 'suckerpunch', 'uturn',
    'vacuumwave', 'waterspout',
}

HAZARDS = {'spikes', 'stealthrock', 'toxicspikes'}

MOVE_PAIRS = [
    ('lightscreen', 'reflect'),
    ('sleeptalk', 'rest'),
    ('protect', 'wish'),
    ('leechseed', 'substitute'),
    ('focuspunch', 'substitute'),
    ('raindance', 'rest'),
]

PRIORITY_POKEMON = {
    'cacturne', 'dusknoir', 'honchkrow', 'mamoswine', 'scizor',
    'shedinja', 'shiftry',
}

STATUS_INFLICTING = {
    'hypnosis', 'stunspore', 'thunderwave', 'toxic', 'willowisp', 'yawn',
}

# Incompatible move pairs (ported from teams.ts)
# Each entry is (set_a, set_b) — if a move from set_a is chosen,
# moves from set_b should be removed from the pool and vice versa.
INCOMPATIBLE_PAIRS: List[Tuple] = [
    # Status moves don't mesh with these
    ('__status__', {'healingwish', 'switcheroo', 'trick'}),
    (SETUP, {'uturn'}),
    (SETUP, HAZARDS),
    (SETUP, {'pursuit', 'toxic'}),
    (PHYSICAL_SETUP, PHYSICAL_SETUP),
    ({'fakeout', 'uturn'}, {'switcheroo', 'trick'}),
    ({'substitute'}, {'uturn'}),
    ({'rest'}, {'substitute'}),
    ({'explosion'}, {'destinybond', 'painsplit', 'rest', 'trick'}),
    # Redundant attacks
    ({'surf'}, {'hydropump'}),
    ({'bodyslam', 'return'}, {'bodyslam', 'doubleedge'}),
    ({'energyball', 'leafstorm'}, {'leafblade', 'leafstorm', 'powerwhip'}),
    ({'lavaplume'}, {'fireblast'}),
    ({'closecombat'}, {'drainpunch'}),
    ({'discharge'}, {'thunderbolt'}),
    ({'gunkshot'}, {'poisonjab'}),
    ({'payback'}, {'pursuit'}),
    ({'protect'}, {'swordsdance'}),
    ({'flamethrower'}, {'overheat'}),
    ({'encore'}, {'roar'}),
    ({'explosion'}, {'whirlwind'}),
    ({'switcheroo'}, {'suckerpunch'}),
    ({'bodyslam'}, {'healingwish'}),
    ({'agility'}, {'vacuumwave'}),
]

# Species incompatibility pairs for team building
SPECIES_INCOMPATIBILITY = [
    ({'parasect', 'toxicroak'}, {'groudon'}),
    ({'shedinja'}, {'tyranitar', 'hippowdon', 'abomasnow'}),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_name(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', name.lower())


def compute_hp(base: int, iv: int, ev: int, level: int) -> int:
    if base == 1:  # Shedinja
        return 1
    return int((2 * base + iv + int(ev / 4)) * level / 100) + level + 10


def compute_stat(base: int, iv: int, ev: int, level: int, nature_mult: float = 1.0) -> int:
    return int((int((2 * base + iv + int(ev / 4)) * level / 100) + 5) * nature_mult)


def build_normalized_lookup(name_to_id: dict) -> dict:
    lookup = {}
    for k, v in name_to_id.items():
        lookup[k] = v
        lookup[normalize_name(k)] = v
    return lookup


def find_id(name: str, lookup: dict) -> int:
    if name in lookup:
        return lookup[name]
    norm = normalize_name(name)
    if norm in lookup:
        return lookup[norm]
    return 0


def fast_pop(lst: list, idx: int):
    """Remove element at idx by swapping with last (O(1))."""
    if idx < 0 or idx >= len(lst):
        return
    lst[idx] = lst[-1]
    lst.pop()


def sample(lst: list):
    """Pick a random element."""
    return lst[random.randrange(len(lst))]


def sample_no_replace(lst: list):
    """Pick and remove a random element."""
    idx = random.randrange(len(lst))
    val = lst[idx]
    fast_pop(lst, idx)
    return val


# ---------------------------------------------------------------------------
# Type effectiveness (Gen 4 chart, no Fairy)
# ---------------------------------------------------------------------------

def build_type_chart() -> Dict[Tuple[str, str], float]:
    """Build Gen 4 type effectiveness lookup."""
    chart = {}
    all_types = ['Normal', 'Fire', 'Water', 'Electric', 'Grass', 'Ice',
                 'Fighting', 'Poison', 'Ground', 'Flying', 'Psychic', 'Bug',
                 'Rock', 'Ghost', 'Dragon', 'Dark', 'Steel']

    # Default: neutral
    for a in all_types:
        for d in all_types:
            chart[(a, d)] = 1.0

    super_effective = [
        ('Fire', 'Grass'), ('Fire', 'Ice'), ('Fire', 'Bug'), ('Fire', 'Steel'),
        ('Water', 'Fire'), ('Water', 'Ground'), ('Water', 'Rock'),
        ('Electric', 'Water'), ('Electric', 'Flying'),
        ('Grass', 'Water'), ('Grass', 'Ground'), ('Grass', 'Rock'),
        ('Ice', 'Grass'), ('Ice', 'Ground'), ('Ice', 'Flying'), ('Ice', 'Dragon'),
        ('Fighting', 'Normal'), ('Fighting', 'Ice'), ('Fighting', 'Rock'),
        ('Fighting', 'Dark'), ('Fighting', 'Steel'),
        ('Poison', 'Grass'),
        ('Ground', 'Fire'), ('Ground', 'Electric'), ('Ground', 'Poison'),
        ('Ground', 'Rock'), ('Ground', 'Steel'),
        ('Flying', 'Grass'), ('Flying', 'Fighting'), ('Flying', 'Bug'),
        ('Psychic', 'Fighting'), ('Psychic', 'Poison'),
        ('Bug', 'Grass'), ('Bug', 'Psychic'), ('Bug', 'Dark'),
        ('Rock', 'Fire'), ('Rock', 'Ice'), ('Rock', 'Flying'), ('Rock', 'Bug'),
        ('Ghost', 'Psychic'), ('Ghost', 'Ghost'),
        ('Dragon', 'Dragon'),
        ('Dark', 'Psychic'), ('Dark', 'Ghost'),
        ('Steel', 'Ice'), ('Steel', 'Rock'),
    ]
    not_very = [
        ('Normal', 'Rock'), ('Normal', 'Steel'),
        ('Fire', 'Fire'), ('Fire', 'Water'), ('Fire', 'Rock'), ('Fire', 'Dragon'),
        ('Water', 'Water'), ('Water', 'Grass'), ('Water', 'Dragon'),
        ('Electric', 'Electric'), ('Electric', 'Grass'), ('Electric', 'Dragon'),
        ('Grass', 'Fire'), ('Grass', 'Grass'), ('Grass', 'Poison'),
        ('Grass', 'Flying'), ('Grass', 'Bug'), ('Grass', 'Dragon'), ('Grass', 'Steel'),
        ('Ice', 'Water'), ('Ice', 'Ice'), ('Ice', 'Fire'), ('Ice', 'Steel'),
        ('Fighting', 'Psychic'), ('Fighting', 'Bug'), ('Fighting', 'Flying'), ('Fighting', 'Poison'),
        ('Poison', 'Poison'), ('Poison', 'Ground'), ('Poison', 'Rock'), ('Poison', 'Ghost'),
        ('Ground', 'Grass'), ('Ground', 'Bug'),
        ('Flying', 'Electric'), ('Flying', 'Rock'), ('Flying', 'Steel'),
        ('Psychic', 'Psychic'), ('Psychic', 'Steel'),
        ('Bug', 'Fire'), ('Bug', 'Fighting'), ('Bug', 'Flying'),
        ('Bug', 'Ghost'), ('Bug', 'Steel'), ('Bug', 'Poison'),
        ('Rock', 'Fighting'), ('Rock', 'Ground'), ('Rock', 'Steel'),
        ('Ghost', 'Dark'),
        ('Dragon', 'Steel'),
        ('Dark', 'Fighting'), ('Dark', 'Dark'),
        ('Steel', 'Fire'), ('Steel', 'Water'), ('Steel', 'Electric'), ('Steel', 'Steel'),
    ]
    immune = [
        ('Normal', 'Ghost'), ('Fighting', 'Ghost'),
        ('Electric', 'Ground'), ('Ground', 'Flying'),
        ('Psychic', 'Dark'), ('Ghost', 'Normal'),
        ('Poison', 'Steel'),
    ]

    for a, d in super_effective:
        chart[(a, d)] = 2.0
    for a, d in not_very:
        chart[(a, d)] = 0.5
    for a, d in immune:
        chart[(a, d)] = 0.0

    return chart


def get_effectiveness(atk_type: str, def_types: List[str],
                      chart: Dict[Tuple[str, str], float]) -> float:
    """Get total effectiveness multiplier (product over defender types).
    Returns log2-style: >0 means weak, <0 means resistant, 0 means neutral."""
    mult = 1.0
    for dt in def_types:
        mult *= chart.get((atk_type, dt), 1.0)
    if mult == 0:
        return -3  # immune
    import math
    return math.log2(mult)


# ---------------------------------------------------------------------------
# Move counter (simplified version of Showdown's MoveCounter)
# ---------------------------------------------------------------------------

class MoveCounter:
    """Tracks move categories/types for the current set."""

    def __init__(self, moves: Set[str], move_data: dict):
        self.physical = 0
        self.special = 0
        self.status = 0
        self.type_count: Dict[str, int] = {}
        self.stab = 0
        self.damaging_moves: Set[str] = set()
        self.priority_count = 0
        self.recoil = 0
        self.skilllink = 0
        self.recovery = 0
        self.setup = 0
        self.hazards = 0
        self._update(moves, move_data)

    def _update(self, moves: Set[str], move_data: dict):
        for m in moves:
            md = move_data.get(m)
            if md is None:
                continue
            cat = md['category']
            mtype = md['type']
            bp = md['basePower']
            prio = md['priority']

            if cat == 'Physical':
                self.physical += 1
            elif cat == 'Special':
                self.special += 1
            else:
                self.status += 1

            if bp > 0 or m in ('seismictoss', 'nightshade', 'counter',
                               'mirrorcoat', 'metalburst'):
                self.damaging_moves.add(m)
                if mtype not in self.type_count:
                    self.type_count[mtype] = 0
                self.type_count[mtype] += 1

            if prio > 0 and bp > 0:
                self.priority_count += 1

            if m in RECOVERY_MOVES:
                self.recovery += 1
            if m in SETUP:
                self.setup += 1
            if m in HAZARDS:
                self.hazards += 1
            if md.get('recoil', False):
                self.recoil += 1
            if md.get('multihit', False):
                self.skilllink += 1

    def get(self, key: str) -> int:
        key_low = key.lower()
        if key_low == 'physical':
            return self.physical
        if key_low == 'special':
            return self.special
        if key_low == 'status':
            return self.status
        if key_low == 'stab':
            return self.stab
        if key_low == 'priority':
            return self.priority_count
        if key_low == 'recoil':
            return self.recoil
        if key_low == 'skilllink':
            return self.skilllink
        if key_low == 'recovery':
            return self.recovery
        if key_low == 'setup':
            return self.setup
        if key_low == 'hazards':
            return self.hazards
        # Type lookup
        return self.type_count.get(key, 0)


def new_query_moves(moves: Set[str], species_types: List[str],
                    preferred_type: str, abilities: List[str],
                    move_data: dict) -> MoveCounter:
    counter = MoveCounter(moves, move_data)
    # Count STAB and preferred
    for m in moves:
        md = move_data.get(m)
        if md is None:
            continue
        mtype = get_move_type(m, md, abilities, preferred_type)
        bp = md['basePower']
        if bp > 0 or m in ('seismictoss', 'nightshade'):
            if mtype in species_types and m not in NO_STAB:
                counter.stab += 1
            if mtype == preferred_type:
                counter.type_count.setdefault('preferred', 0)
                counter.type_count['preferred'] = counter.type_count.get('preferred', 0) + 1
    return counter


def get_move_type(move_id: str, md: dict, abilities: List[str],
                  preferred_type: str) -> str:
    """Get effective move type, handling Hidden Power and Normalize etc."""
    if move_id.startswith('hiddenpower'):
        hp_type = move_id[11:]
        # Capitalize first letter
        return hp_type.capitalize() if hp_type else md['type']
    if 'Normalize' in abilities:
        return 'Normal'
    return md['type']


# ---------------------------------------------------------------------------
# Move selection logic (ported from gen4/teams.ts)
# ---------------------------------------------------------------------------

def get_status_moves(move_data: dict) -> Set[str]:
    """Get all status category moves."""
    return {m for m, d in move_data.items() if d['category'] == 'Status'}


def incompatible_moves(moves: Set[str], move_pool: list,
                       a, b):
    """If any move from set A is in moves, remove all of set B from pool,
    and vice versa."""
    if isinstance(a, str):
        a = {a}
    if isinstance(b, str):
        b = {b}
    a = set(a)
    b = set(b)

    a_in_moves = bool(moves & a)
    b_in_moves = bool(moves & b)

    if a_in_moves:
        for m in list(b):
            if m in move_pool:
                move_pool.remove(m)
    if b_in_moves:
        for m in list(a):
            if m in move_pool:
                move_pool.remove(m)


def cull_move_pool(types: List[str], moves: Set[str], abilities: List[str],
                   counter: MoveCounter, move_pool: list,
                   team_details: dict, species_id: str,
                   is_lead: bool, preferred_type: str, role: str,
                   move_data: dict, status_moves: Set[str]):
    """Port of RandomGen4Teams.cullMovePool."""
    max_move_count = 4

    # No duplicate Hidden Powers
    has_hp = any(m.startswith('hiddenpower') for m in moves)
    if has_hp:
        move_pool[:] = [m for m in move_pool if not m.startswith('hiddenpower')]

    if len(moves) + len(move_pool) <= max_move_count:
        return

    # If two unfilled moves and only one unpaired move, cull the unpaired
    if len(moves) == max_move_count - 2:
        unpaired = list(move_pool)
        for p0, p1 in MOVE_PAIRS:
            if p0 in move_pool and p1 in move_pool:
                if p0 in unpaired:
                    unpaired.remove(p0)
                if p1 in unpaired:
                    unpaired.remove(p1)
        if len(unpaired) == 1:
            if unpaired[0] in move_pool:
                move_pool.remove(unpaired[0])

    # Paired moves: if only 1 slot left, remove pairs that need 2 slots
    if len(moves) == max_move_count - 1:
        for p0, p1 in MOVE_PAIRS:
            if p0 in move_pool and p1 in move_pool:
                if p0 in move_pool:
                    move_pool.remove(p0)
                if p1 in move_pool:
                    move_pool.remove(p1)

    # Team-based culls
    if team_details.get('screens') and len(move_pool) >= max_move_count + 2:
        if 'reflect' in move_pool:
            move_pool.remove('reflect')
        if 'lightscreen' in move_pool:
            move_pool.remove('lightscreen')
        if len(moves) + len(move_pool) <= max_move_count:
            return

    if team_details.get('stealthRock'):
        if 'stealthrock' in move_pool:
            move_pool.remove('stealthrock')
        if len(moves) + len(move_pool) <= max_move_count:
            return

    if team_details.get('rapidSpin'):
        if 'rapidspin' in move_pool:
            move_pool.remove('rapidspin')
        if len(moves) + len(move_pool) <= max_move_count:
            return

    if team_details.get('toxicSpikes'):
        if 'toxicspikes' in move_pool:
            move_pool.remove('toxicspikes')
        if len(moves) + len(move_pool) <= max_move_count:
            return

    if team_details.get('spikes', 0) >= 2:
        if 'spikes' in move_pool:
            move_pool.remove('spikes')
        if len(moves) + len(move_pool) <= max_move_count:
            return

    if team_details.get('statusCure'):
        if 'aromatherapy' in move_pool:
            move_pool.remove('aromatherapy')
        if 'healbell' in move_pool:
            move_pool.remove('healbell')
        if len(moves) + len(move_pool) <= max_move_count:
            return

    # General incompatibilities
    bad_with_setup = {'pursuit', 'toxic'}

    for pair in INCOMPATIBLE_PAIRS:
        a_set, b_set = pair
        if a_set == '__status__':
            a_set = status_moves
        incompatible_moves(moves, move_pool, a_set, b_set)

    # Status-inflicting moves shouldn't stack (unless Staller)
    if role != 'Staller':
        incompatible_moves(moves, move_pool, STATUS_INFLICTING, STATUS_INFLICTING)

    # Bastiodon special case
    if species_id == 'bastiodon':
        incompatible_moves(moves, move_pool,
                           {'metalburst', 'protect', 'roar'},
                           {'metalburst', 'protect'})


def run_enforcement_checker(type_name: str, move_pool: list, moves: Set[str],
                            abilities: List[str], types: List[str],
                            counter: MoveCounter, species_id: str,
                            base_stats: dict, team_details: dict) -> bool:
    """Port of moveEnforcementCheckers from gen4/teams.ts."""
    types_set = set(types)
    b_atk = base_stats.get('attack', base_stats.get('atk', 80))

    checks = {
        'Bug': lambda: not counter.get('Bug') and 'megahorn' in move_pool,
        'Dark': lambda: not counter.get('Dark'),
        'Dragon': lambda: not counter.get('Dragon'),
        'Electric': lambda: not counter.get('Electric'),
        'Fighting': lambda: not counter.get('Fighting'),
        'Fire': lambda: not counter.get('Fire'),
        'Flying': lambda: not counter.get('Flying') and species_id != 'aerodactyl',
        'Ghost': lambda: not counter.get('Ghost'),
        'Grass': lambda: (not counter.get('Grass') and
                          (b_atk >= 100 or 'leafstorm' in move_pool or 'solarbeam' in move_pool)),
        'Ground': lambda: not counter.get('Ground'),
        'Ice': lambda: not counter.get('Ice'),
        'Poison': lambda: (not counter.get('Poison') and
                           bool(types_set & {'Ghost', 'Grass', 'Ground'})),
        'Psychic': lambda: (not counter.get('Psychic') and
                            ('Fighting' in types_set or 'calmmind' in move_pool)),
        'Rock': lambda: not counter.get('Rock') and b_atk >= 80,
        'Steel': lambda: not counter.get('Steel') and species_id == 'metagross',
        'Water': lambda: not counter.get('Water'),
    }

    if type_name in checks:
        return checks[type_name]()
    return False


def add_move(move_id: str, moves: Set[str], move_pool: list):
    """Add a move to the set and remove from pool."""
    moves.add(move_id)
    if move_id in move_pool:
        move_pool.remove(move_id)


def random_moveset(types: List[str], abilities: List[str],
                   team_details: dict, species_id: str,
                   base_stats: dict, is_lead: bool,
                   move_pool: list, preferred_type: str, role: str,
                   move_data: dict, status_moves: Set[str]) -> Set[str]:
    """Port of RandomGen4Teams.randomMoveset."""
    moves: Set[str] = set()
    counter = new_query_moves(moves, types, preferred_type, abilities, move_data)
    max_move_count = 4

    cull_move_pool(types, moves, abilities, counter, move_pool, team_details,
                   species_id, is_lead, preferred_type, role, move_data,
                   status_moves)

    # If 4 or fewer moves left, take them all
    if len(move_pool) <= max_move_count:
        while move_pool:
            m = sample(move_pool)
            add_move(m, moves, move_pool)
        return moves

    def refresh_counter():
        return new_query_moves(moves, types, preferred_type, abilities, move_data)

    # Enforce Facade if Guts is a possible ability
    if 'facade' in move_pool and 'Guts' in abilities:
        add_move('facade', moves, move_pool)
        counter = refresh_counter()

    # Enforce Seismic Toss, Spore, Volt Tackle
    for m in ('seismictoss', 'spore', 'volttackle'):
        if m in move_pool:
            add_move(m, moves, move_pool)
            counter = refresh_counter()

    # Enforce Substitute on non-Setup sets with Baton Pass
    if 'Setup' not in role:
        if 'batonpass' in move_pool and 'substitute' in move_pool:
            add_move('substitute', moves, move_pool)
            counter = refresh_counter()

    # Enforce Rapid Spin for Bulky Support / Spinner if team lacks it
    if role in ('Bulky Support', 'Spinner') and not team_details.get('rapidSpin'):
        if 'rapidspin' in move_pool:
            add_move('rapidspin', moves, move_pool)
            counter = refresh_counter()

    # Enforce STAB priority for Bulky Attacker / Bulky Setup / priority mons
    if role in ('Bulky Attacker', 'Bulky Setup') or species_id in PRIORITY_POKEMON:
        prio_moves = []
        for m in move_pool:
            md = move_data.get(m)
            if md and md['priority'] > 0 and (md['basePower'] > 0):
                mtype = get_move_type(m, md, abilities, preferred_type)
                if mtype in types:
                    prio_moves.append(m)
        if prio_moves:
            add_move(sample(prio_moves), moves, move_pool)
            counter = refresh_counter()

    # Enforce STAB
    for t in types:
        stab_moves = []
        for m in move_pool:
            md = move_data.get(m)
            if md and m not in NO_STAB and (md['basePower'] > 0):
                mtype = get_move_type(m, md, abilities, preferred_type)
                if mtype == t:
                    stab_moves.append(m)
        while (run_enforcement_checker(t, move_pool, moves, abilities, types,
                                       counter, species_id, base_stats,
                                       team_details) and
               len(moves) < max_move_count and stab_moves):
            idx = random.randrange(len(stab_moves))
            m = stab_moves[idx]
            stab_moves[idx] = stab_moves[-1]
            stab_moves.pop()
            add_move(m, moves, move_pool)
            counter = refresh_counter()

    # Enforce preferred type
    if not counter.get('preferred'):
        pref_moves = []
        for m in move_pool:
            md = move_data.get(m)
            if md and m not in NO_STAB and (md['basePower'] > 0):
                mtype = get_move_type(m, md, abilities, preferred_type)
                if mtype == preferred_type:
                    pref_moves.append(m)
        if pref_moves and len(moves) < max_move_count:
            add_move(sample(pref_moves), moves, move_pool)
            counter = refresh_counter()

    # If no STAB move, add one
    if not counter.stab and len(moves) < max_move_count:
        stab_moves = []
        for m in move_pool:
            md = move_data.get(m)
            if md and m not in NO_STAB and (md['basePower'] > 0):
                mtype = get_move_type(m, md, abilities, preferred_type)
                if mtype in types:
                    stab_moves.append(m)
        if stab_moves:
            add_move(sample(stab_moves), moves, move_pool)
            counter = refresh_counter()
        elif 'uturn' in move_pool and 'Bug' in types:
            add_move('uturn', moves, move_pool)
            counter = refresh_counter()

    # Enforce Stealth Rock if team doesn't have it
    if 'stealthrock' in move_pool and not team_details.get('stealthRock') and len(moves) < max_move_count:
        add_move('stealthrock', moves, move_pool)
        counter = refresh_counter()

    # Enforce recovery for bulky roles
    if role in ('Bulky Support', 'Bulky Attacker', 'Bulky Setup', 'Spinner', 'Staller'):
        rec_moves = [m for m in move_pool if m in RECOVERY_MOVES]
        if rec_moves and len(moves) < max_move_count:
            add_move(sample(rec_moves), moves, move_pool)
            counter = refresh_counter()

    # Enforce Staller moves
    if role == 'Staller':
        for m in ('protect', 'toxic', 'wish'):
            if m in move_pool and len(moves) < max_move_count:
                add_move(m, moves, move_pool)
                counter = refresh_counter()

    # Enforce setup for Setup roles
    if 'Setup' in role:
        setup_moves = [m for m in move_pool if m in SETUP]
        if setup_moves and len(moves) < max_move_count:
            add_move(sample(setup_moves), moves, move_pool)
            counter = refresh_counter()

    # Enforce at least one attacking move
    if not counter.damaging_moves and not ('uturn' in moves and 'Bug' in types):
        atk_moves = []
        for m in move_pool:
            md = move_data.get(m)
            if md and m not in NO_STAB and md['category'] != 'Status':
                atk_moves.append(m)
        if atk_moves and len(moves) < max_move_count:
            add_move(sample(atk_moves), moves, move_pool)
            counter = refresh_counter()

    # Enforce coverage for attackers
    if role in ('Fast Attacker', 'Setup Sweeper', 'Bulky Attacker', 'Wallbreaker'):
        if len(counter.damaging_moves) == 1 and len(moves) < max_move_count:
            current_type = None
            for m in counter.damaging_moves:
                md = move_data.get(m)
                if md:
                    current_type = get_move_type(m, md, abilities, preferred_type)
            if current_type:
                cov_moves = []
                for m in move_pool:
                    md = move_data.get(m)
                    if md and m not in NO_STAB and (md['basePower'] > 0):
                        mtype = get_move_type(m, md, abilities, preferred_type)
                        if mtype != current_type:
                            cov_moves.append(m)
                if cov_moves:
                    add_move(sample(cov_moves), moves, move_pool)
                    counter = refresh_counter()

    # Fill remaining slots randomly, respecting move pairs
    while len(moves) < max_move_count and move_pool:
        m = sample(move_pool)
        add_move(m, moves, move_pool)
        # Enforce paired moves
        for p0, p1 in MOVE_PAIRS:
            if m == p0 and p1 in move_pool and len(moves) < max_move_count:
                add_move(p1, moves, move_pool)
            if m == p1 and p0 in move_pool and len(moves) < max_move_count:
                add_move(p0, moves, move_pool)

    return moves


# ---------------------------------------------------------------------------
# Ability selection (ported from gen4/teams.ts)
# ---------------------------------------------------------------------------

def should_cull_ability(ability: str, types: List[str], moves: Set[str],
                        counter: MoveCounter, team_details: dict) -> bool:
    if ability == 'Chlorophyll':
        return not team_details.get('sun')
    if ability == 'Swift Swim':
        return not team_details.get('rain')
    if ability == 'Rock Head':
        return counter.recoil == 0
    if ability == 'Skill Link':
        return counter.skilllink == 0
    return False


def get_ability(types: List[str], moves: Set[str], abilities: List[str],
                counter: MoveCounter, team_details: dict,
                species_id: str) -> str:
    if len(abilities) <= 1:
        return abilities[0] if abilities else ''

    # Hard-coded species
    if species_id == 'dewgong':
        return 'Hydration' if 'raindance' in moves else 'Thick Fat'
    if species_id == 'cloyster' and counter.skilllink > 0:
        return 'Skill Link'

    allowed = [a for a in abilities
               if not should_cull_ability(a, types, moves, counter, team_details)]

    if allowed:
        return sample(allowed)

    # If all culled, prefer weather abilities
    weather_abs = [a for a in abilities if a in ('Chlorophyll', 'Swift Swim')]
    if weather_abs:
        return sample(weather_abs)

    return sample(abilities)


# ---------------------------------------------------------------------------
# Item selection (ported from gen4/teams.ts)
# ---------------------------------------------------------------------------

def get_priority_item(ability: str, types: List[str], moves: Set[str],
                      counter: MoveCounter, team_details: dict,
                      species_id: str, is_lead: bool,
                      preferred_type: str, role: str,
                      base_stats: dict) -> Optional[str]:
    b_spe = base_stats.get('speed', base_stats.get('spe', 80))
    b_spa = base_stats.get('special-attack', base_stats.get('spa', 80))
    b_atk = base_stats.get('attack', base_stats.get('atk', 80))

    if species_id in ('latias', 'latios'):
        return 'Soul Dew'
    if species_id == 'marowak':
        return 'Thick Club'
    if species_id == 'pikachu':
        return 'Light Ball'
    if species_id in ('shedinja', 'smeargle'):
        return 'Focus Sash'
    if species_id == 'unown':
        return 'Choice Specs'
    if species_id == 'wobbuffet':
        return 'Custap Berry'
    if species_id == 'ditto' or (species_id == 'rampardos' and role == 'Fast Attacker'):
        return 'Choice Scarf'
    if species_id == 'honchkrow':
        return 'Life Orb'
    if ability == 'Poison Heal' or 'facade' in moves:
        return 'Toxic Orb'
    if ability == 'Speed Boost' and species_id == 'yanmega':
        return 'Life Orb'
    if moves & {'healingwish', 'switcheroo', 'trick'}:
        if 60 <= b_spe <= 108 and role != 'Wallbreaker' and not counter.priority_count:
            return 'Choice Scarf'
        else:
            return 'Choice Band' if counter.physical > counter.special else 'Choice Specs'
    if 'bellydrum' in moves:
        return 'Sitrus Berry'
    if 'waterspout' in moves:
        return 'Choice Scarf'
    if ability == 'Magic Guard':
        return 'Life Orb'
    if 'lightscreen' in moves and 'reflect' in moves:
        return 'Light Clay'
    if 'rest' in moves and 'sleeptalk' not in moves and ability not in ('Natural Cure', 'Shed Skin'):
        if 'raindance' in moves and ability == 'Hydration':
            return 'Damp Rock'
        return 'Chesto Berry'
    if ability == 'Unburden':
        return 'Sitrus Berry'
    if role == 'Staller':
        return 'Leftovers'
    return None


def get_item(ability: str, types: List[str], moves: Set[str],
             counter: MoveCounter, team_details: dict,
             species_id: str, is_lead: bool,
             preferred_type: str, role: str,
             base_stats: dict, type_chart: dict) -> str:
    b_spe = base_stats.get('speed', base_stats.get('spe', 80))
    b_spa = base_stats.get('special-attack', base_stats.get('spa', 80))
    b_atk = base_stats.get('attack', base_stats.get('atk', 80))
    b_hp = base_stats.get('hp', 80)
    b_def = base_stats.get('defense', base_stats.get('def', 80))
    b_spd = base_stats.get('special-defense', base_stats.get('spd', 80))
    def_total = b_hp + b_def + b_spd

    scarf_reqs = (role != 'Wallbreaker' and 60 <= b_spe <= 108 and
                  not counter.priority_count and 'pursuit' not in moves)

    rock_eff = get_effectiveness('Rock', types, type_chart)

    # Dark-type attacker with pursuit + sucker punch
    if ('pursuit' in moves and 'suckerpunch' in moves and counter.get('Dark') and
            (species_id not in PRIORITY_POKEMON or counter.get('Dark') >= 2)):
        return 'Black Glasses'

    if counter.special == 4:
        if scarf_reqs and b_spa >= 90 and random.random() < 0.5:
            return 'Choice Scarf'
        return 'Choice Specs'

    if counter.special == 3 and role == 'Fast Attacker' and moves & {'explosion', 'selfdestruct'}:
        return 'Choice Scarf'
    if counter.special == 3 and 'uturn' in moves:
        return 'Choice Specs'

    if (counter.physical == 4 and species_id != 'jirachi' and
            not (moves & {'fakeout', 'rapidspin'})):
        if (scarf_reqs and (b_atk >= 100 or ability in ('Pure Power', 'Huge Power')) and
                random.random() < 0.5):
            return 'Choice Scarf'
        return 'Choice Band'

    if 'Normal' in types and 'fakeout' in moves and counter.get('Normal'):
        return 'Silk Scarf'
    if species_id == 'palkia':
        return 'Lustrous Orb'
    if species_id == 'farfetchd':
        return 'Stick'
    if 'outrage' in moves and counter.setup and 'sleeptalk' not in moves:
        return 'Lum Berry'
    if moves & {'batonpass', 'protect', 'substitute'}:
        return 'Leftovers'
    if (role == 'Fast Support' and is_lead and def_total < 255 and
            not counter.recovery and (counter.hazards or counter.setup) and
            (not counter.recoil or ability == 'Rock Head')):
        return 'Focus Sash'

    # Default items
    if role == 'Fast Support':
        if (counter.physical + counter.special >= 3 and
                not (moves & {'rapidspin', 'uturn'}) and rock_eff < 2):
            return 'Life Orb'
        return 'Leftovers'

    # Expert Belt check
    no_expert_types = {'Dragon', 'Normal', 'Poison'}
    has_no_expert_type = bool(set(counter.type_count.keys()) & no_expert_types)
    expert_belt_ok = not has_no_expert_type

    if (not counter.status and expert_belt_ok and
            ('uturn' in moves or role == 'Fast Attacker')):
        return 'Expert Belt'

    if (role in ('Fast Attacker', 'Setup Sweeper', 'Wallbreaker') and
            rock_eff < 2 and 'rapidspin' not in moves):
        return 'Life Orb'

    return 'Leftovers'


# ---------------------------------------------------------------------------
# Full set generation (ported from gen4/teams.ts randomSet)
# ---------------------------------------------------------------------------

def random_set(species_id: str, sets_data: dict, base_stats_all: dict,
               team_details: dict, is_lead: bool,
               move_data: dict, status_moves: Set[str],
               move_lookup: dict, ability_lookup: dict, item_lookup: dict,
               moves_np, type_chart: dict) -> Optional[dict]:
    """Generate a full random set for a species. Returns None on failure."""
    sdata = sets_data.get(species_id)
    if sdata is None:
        return None

    level = sdata.get('level', 80)
    all_sets = sdata.get('sets', [])
    if not all_sets:
        return None

    # Pick role — enforce Spinner if team lacks Rapid Spin
    can_spinner = any(s['role'] == 'Spinner' for s in all_sets) and not team_details.get('rapidSpin')
    possible_sets = []
    for s in all_sets:
        if team_details.get('rapidSpin') and s['role'] == 'Spinner':
            continue
        if can_spinner and s['role'] != 'Spinner':
            continue
        possible_sets.append(s)

    if not possible_sets:
        possible_sets = all_sets

    chosen_set = sample(possible_sets)
    role = chosen_set['role']
    move_pool = list(chosen_set.get('movepool', []))
    abilities = list(chosen_set.get('abilities', ['']))
    preferred_types = chosen_set.get('preferredTypes', [])
    preferred_type = sample(preferred_types) if preferred_types else ''

    # Get base stats
    bs = base_stats_all.get(species_id, {})
    types = bs.get('types', ['Normal'])

    # Generate moveset
    moves = random_moveset(types, abilities, team_details, species_id,
                           bs, is_lead, move_pool, preferred_type, role,
                           move_data, status_moves)

    counter = new_query_moves(moves, types, preferred_type, abilities, move_data)

    # Get ability
    ability = get_ability(types, moves, abilities, counter, team_details, species_id)

    # Get item
    item = get_priority_item(ability, types, moves, counter, team_details,
                             species_id, is_lead, preferred_type, role, bs)
    if item is None:
        item = get_item(ability, types, moves, counter, team_details,
                        species_id, is_lead, preferred_type, role, bs, type_chart)

    # Poison type with Leftovers → Black Sludge
    if item == 'Leftovers' and 'Poison' in types:
        item = 'Black Sludge'

    # EV/IV optimization
    evs = {'hp': 85, 'atk': 85, 'def': 85, 'spa': 85, 'spd': 85, 'spe': 85}
    ivs = {'hp': 31, 'atk': 31, 'def': 31, 'spa': 31, 'spd': 31, 'spe': 31}

    # Handle Hidden Power IVs
    has_hp = False
    hp_type = ''
    for m in moves:
        if m.startswith('hiddenpower'):
            has_hp = True
            hp_type = m[11:]

    HP_IVS = {
        'bug': {'atk': 30, 'def': 30, 'spd': 30},
        'dark': {},
        'dragon': {'atk': 30},
        'electric': {'spa': 30},
        'fighting': {'def': 30, 'spa': 30, 'spd': 30, 'spe': 30},
        'fire': {'atk': 30, 'spa': 30, 'spe': 30},
        'flying': {'hp': 30, 'atk': 30, 'def': 30, 'spa': 30, 'spd': 30},
        'ghost': {'def': 30, 'spd': 30},
        'grass': {'atk': 30, 'spa': 30},
        'ground': {'spa': 30, 'spd': 30},
        'ice': {'atk': 30, 'def': 30},
        'poison': {'def': 30, 'spa': 30, 'spd': 30},
        'psychic': {'atk': 30, 'spe': 30},
        'rock': {'def': 30, 'spd': 30, 'spe': 30},
        'steel': {'spd': 30},
        'water': {'atk': 30, 'def': 30, 'spa': 30},
    }
    if has_hp and hp_type in HP_IVS:
        for stat, val in HP_IVS[hp_type].items():
            ivs[stat] = val

    # Optimize HP EVs for Stealth Rock
    sr_immunity = ability == 'Magic Guard'
    sr_weakness = 0 if sr_immunity else get_effectiveness('Rock', types, type_chart)
    b_hp = bs.get('hp', 80)

    while evs['hp'] > 1:
        hp = int((2 * b_hp + ivs['hp'] + int(evs['hp'] / 4)) * level / 100) + level + 10
        if 'substitute' in moves and item == 'Sitrus Berry':
            if hp % 4 == 0:
                break
        elif 'bellydrum' in moves and item == 'Sitrus Berry':
            if hp % 2 == 0:
                break
        else:
            if sr_weakness <= 0:
                break
            if sr_weakness == 1 and item in ('Black Sludge', 'Leftovers', 'Life Orb'):
                break
            if item != 'Sitrus Berry' and sr_weakness > 0:
                divisor = 4 / sr_weakness if sr_weakness > 0 else 4
                if divisor > 0 and hp % divisor > 0:
                    break
            if item == 'Sitrus Berry' and sr_weakness > 0:
                divisor = 4 / sr_weakness if sr_weakness > 0 else 4
                if divisor > 0 and hp % divisor == 0:
                    break
        evs['hp'] -= 4

    # Minimize confusion damage for special attackers
    if not counter.physical and 'transform' not in moves:
        evs['atk'] = 0
        ivs['atk'] = (ivs['atk'] - 28) if has_hp and ivs['atk'] > 28 else 0

    # Minimize speed for Gyro Ball / Trick Room
    if moves & {'gyroball', 'metalburst', 'trickroom'}:
        evs['spe'] = 0
        ivs['spe'] = (ivs['spe'] - 28) if has_hp and ivs['spe'] > 28 else 0

    # Compute final max HP
    max_hp = compute_hp(b_hp, ivs['hp'], evs['hp'], level)

    return {
        'species_id': species_id,
        'level': level,
        'moves': list(moves),
        'ability': ability,
        'item': item,
        'types': types,
        'base_stats': bs,
        'evs': evs,
        'ivs': ivs,
        'max_hp': max_hp,
        'role': role,
    }


# ---------------------------------------------------------------------------
# Full team generation (ported from gen5/teams.ts randomTeam)
# ---------------------------------------------------------------------------

def random_team(sets_data: dict, base_stats_all: dict,
                move_data: dict, status_moves: Set[str],
                move_lookup: dict, ability_lookup: dict, item_lookup: dict,
                moves_np, type_chart: dict) -> Optional[List[dict]]:
    """Generate a full 6-Pokemon team matching Showdown's constraints."""
    pokemon = []
    base_formes = set()
    type_count: Dict[str, int] = {}
    type_weaknesses: Dict[str, float] = {}
    type_double_weaknesses: Dict[str, float] = {}
    team_details: dict = {}
    num_max_level = 0

    all_types = ['Normal', 'Fire', 'Water', 'Electric', 'Grass', 'Ice',
                 'Fighting', 'Poison', 'Ground', 'Flying', 'Psychic', 'Bug',
                 'Rock', 'Ghost', 'Dragon', 'Dark', 'Steel']

    species_pool = list(sets_data.keys())
    random.shuffle(species_pool)

    attempts = 0
    max_attempts = len(species_pool) * 3

    i = 0
    while i < len(species_pool) and len(pokemon) < 6 and attempts < max_attempts:
        species_id = species_pool[i]
        i += 1
        attempts += 1

        # Species clause
        if species_id in base_formes:
            continue

        bs = base_stats_all.get(species_id, {})
        types = bs.get('types', ['Normal'])
        sdata = sets_data.get(species_id, {})
        level = sdata.get('level', 80)

        # Type balance: max 2 of any type
        skip = False
        for t in types:
            if type_count.get(t, 0) >= 2:
                skip = True
                break
        if skip:
            continue

        # Weakness balance: max 3 weak, max 1 double-weak to any type
        for t in all_types:
            eff = get_effectiveness(t, types, type_chart)
            if eff > 0:
                if type_weaknesses.get(t, 0) >= 3:
                    skip = True
                    break
            if eff > 1:
                if type_double_weaknesses.get(t, 0) >= 1:
                    skip = True
                    break
        if skip:
            continue

        # Dry Skin counted as Fire weakness
        all_abilities = set()
        for s in sdata.get('sets', []):
            all_abilities.update(s.get('abilities', []))
        if (get_effectiveness('Fire', types, type_chart) == 0 and
                'Dry Skin' in all_abilities):
            if type_weaknesses.get('Fire', 0) >= 3:
                continue

        # Limit one level 100 Pokemon
        if level == 100 and num_max_level >= 1:
            continue

        # Species compatibility
        compat_ok = True
        for group_a, group_b in SPECIES_INCOMPATIBILITY:
            if species_id in group_b:
                if any(p['species_id'] in group_a for p in pokemon):
                    compat_ok = False
                    break
            if species_id in group_a:
                if any(p['species_id'] in group_b for p in pokemon):
                    compat_ok = False
                    break
        if not compat_ok:
            continue

        # Generate the set
        poke_set = random_set(species_id, sets_data, base_stats_all,
                              team_details, len(pokemon) == 0,
                              move_data, status_moves,
                              move_lookup, ability_lookup, item_lookup,
                              moves_np, type_chart)
        if poke_set is None:
            continue

        pokemon.append(poke_set)

        if len(pokemon) == 6:
            break

        # Update counters
        base_formes.add(species_id)

        for t in types:
            type_count[t] = type_count.get(t, 0) + 1

        for t in all_types:
            eff = get_effectiveness(t, types, type_chart)
            if eff > 0:
                type_weaknesses[t] = type_weaknesses.get(t, 0) + 1
            if eff > 1:
                type_double_weaknesses[t] = type_double_weaknesses.get(t, 0) + 1

        if poke_set['ability'] == 'Dry Skin' and get_effectiveness('Fire', types, type_chart) == 0:
            type_weaknesses['Fire'] = type_weaknesses.get('Fire', 0) + 1

        if level == 100:
            num_max_level += 1

        # Update team details
        set_moves = set(poke_set['moves'])
        if poke_set['ability'] == 'Snow Warning' or 'hail' in set_moves:
            team_details['hail'] = 1
        if poke_set['ability'] == 'Drizzle' or 'raindance' in set_moves:
            team_details['rain'] = 1
        if poke_set['ability'] == 'Sand Stream':
            team_details['sand'] = 1
        if poke_set['ability'] == 'Drought' or 'sunnyday' in set_moves:
            team_details['sun'] = 1
        if set_moves & {'aromatherapy', 'healbell'}:
            team_details['statusCure'] = 1
        if 'spikes' in set_moves:
            team_details['spikes'] = team_details.get('spikes', 0) + 1
        if 'stealthrock' in set_moves:
            team_details['stealthRock'] = 1
        if 'toxicspikes' in set_moves:
            team_details['toxicSpikes'] = 1
        if 'rapidspin' in set_moves:
            team_details['rapidSpin'] = 1
        if 'reflect' in set_moves and 'lightscreen' in set_moves:
            team_details['screens'] = 1

    if len(pokemon) < 6:
        return None

    return pokemon


# ---------------------------------------------------------------------------
# Build move_data dict from engine tables
# ---------------------------------------------------------------------------

def build_move_data(moves_np, move_lookup: dict) -> dict:
    """Build a move_id_str → {type, category, basePower, priority, ...} dict."""
    rev_lookup = {}
    for name, mid in move_lookup.items():
        norm = normalize_name(name)
        if norm not in rev_lookup or len(name) < len(rev_lookup.get(norm, ('', 0))[0]):
            rev_lookup[norm] = (name, mid)
        # Also store by exact name
        if name not in rev_lookup:
            rev_lookup[name] = (name, mid)

    # Build: normalized_name → data
    data = {}
    for name, mid in move_lookup.items():
        norm = normalize_name(name)
        if mid <= 0 or mid >= len(moves_np):
            continue
        row = moves_np[mid]
        bp = int(row[MF_BASE_POWER])
        mtype_id = int(row[MF_TYPE])
        cat_id = int(row[MF_CATEGORY])
        prio = int(row[MF_PRIORITY])
        # Convert signed priority (stored as int16)
        if prio > 127:
            prio -= 256
        pp = int(row[MF_PP])

        category = ['Physical', 'Special', 'Status'][min(cat_id, 2)]
        mtype = TYPE_MAP_REV.get(mtype_id, 'Normal')

        recoil = int(row[MF_RECOIL_NUM]) != 0
        multihit = int(row[MF_MULTIHIT_MAX]) > 1

        data[norm] = {
            'type': mtype,
            'category': category,
            'basePower': bp,
            'priority': prio,
            'pp': pp,
            'recoil': recoil,
            'multihit': multihit,
            'id': mid,
        }

    return data


# ---------------------------------------------------------------------------
# Convert a pokemon set dict → int16[FIELDS_PER_MON] array row
# ---------------------------------------------------------------------------

def set_to_array(poke_set: dict, species_lookup: dict, move_lookup: dict,
                 ability_lookup: dict, item_lookup: dict,
                 moves_np, base_stats_all: dict) -> np.ndarray:
    """Convert a generated set to the int16 array format."""
    row = np.zeros(FIELDS_PER_MON, dtype=np.int16)

    species_id_str = poke_set['species_id']
    species_id_num = find_id(species_id_str, species_lookup)
    ability_id = find_id(poke_set['ability'], ability_lookup)
    item_id = find_id(poke_set['item'], item_lookup)

    bs = poke_set['base_stats']
    types = poke_set['types']
    level = poke_set['level']

    type1 = TYPE_MAP.get(types[0], 1)
    type2 = TYPE_MAP.get(types[1], 0) if len(types) > 1 else 0

    b_hp = bs.get('hp', 80)
    b_atk = bs.get('attack', bs.get('atk', 80))
    b_def = bs.get('defense', bs.get('def', 80))
    b_spa = bs.get('special-attack', bs.get('spa', 80))
    b_spd = bs.get('special-defense', bs.get('spd', 80))
    b_spe = bs.get('speed', bs.get('spe', 80))

    weight_hg = int(bs.get('weightkg', 80) * 10)

    move_ids = []
    move_pps = []
    move_list = list(poke_set['moves'])
    while len(move_list) < 4:
        move_list.append('')

    for m in move_list[:4]:
        mid = find_id(m, move_lookup)
        move_ids.append(mid)
        if mid > 0 and mid < len(moves_np):
            pp = int(moves_np[mid, MF_PP])
            pp = pp * 8 // 5  # Max PP ups for randbats
            move_pps.append(min(pp, 64))
        else:
            move_pps.append(24)

    row[0] = species_id_num
    row[1] = ability_id
    row[2] = item_id
    row[3] = type1
    row[4] = type2
    row[5] = b_hp
    row[6] = b_atk
    row[7] = b_def
    row[8] = b_spa
    row[9] = b_spd
    row[10] = b_spe
    row[11] = poke_set['max_hp']
    row[12:16] = move_ids
    row[16:20] = move_pps
    row[20] = level
    row[21] = min(weight_hg, 32767)  # int16 max

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate Gen 4 random battle team pool (Showdown-faithful)')
    parser.add_argument('--n-teams', type=int, default=100000,
                        help='Number of teams to generate')
    parser.add_argument('--showdown-dir', type=str,
                        default=str(Path(__file__).resolve().parent.parent.parent /
                                    'PokemonShowdownClaude' / 'pokemon-showdown' /
                                    'data' / 'random-battles' / 'gen4'),
                        help='Path to Showdown gen4 random battles data dir')
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

    # Load engine tables
    print("Loading engine tables...")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pokejax.data.tables import load_tables
    tables = load_tables(4)
    print(f"  Species: {len(tables.species_name_to_id)}, "
          f"Moves: {len(tables.move_name_to_id)}, "
          f"Abilities: {len(tables.ability_name_to_id)}, "
          f"Items: {len(tables.item_name_to_id)}")

    species_lookup = build_normalized_lookup(tables.species_name_to_id)
    move_lookup = build_normalized_lookup(tables.move_name_to_id)
    ability_lookup = build_normalized_lookup(tables.ability_name_to_id)
    item_lookup = build_normalized_lookup(tables.item_name_to_id)

    moves_np = np.array(tables.moves)

    # Load sets.json (the actual Showdown data source)
    sets_path = os.path.join(args.showdown_dir, 'sets.json')
    print(f"Loading sets.json from {sets_path}...")
    with open(sets_path) as f:
        sets_data = json.load(f)
    print(f"  {len(sets_data)} species in sets.json")

    # Load base stats
    base_stats_path = os.path.join(args.bot_data_dir, 'gen4_base_stats.json')
    print(f"Loading base stats from {base_stats_path}...")
    base_stats_all = {}
    if os.path.exists(base_stats_path):
        with open(base_stats_path) as f:
            raw_bs = json.load(f)
        # Normalize keys to match sets.json species IDs
        for k, v in raw_bs.items():
            base_stats_all[normalize_name(k)] = v
    print(f"  {len(base_stats_all)} species in base stats")

    # Validate species mapping
    mapped = 0
    unmapped = []
    for sid in sets_data:
        if find_id(sid, species_lookup) > 0:
            mapped += 1
        else:
            unmapped.append(sid)
    print(f"  Species mapped to engine: {mapped}/{len(sets_data)}")
    if unmapped and len(unmapped) <= 20:
        print(f"  Unmapped: {unmapped}")

    # Build move data dict from engine tables
    print("Building move data lookup...")
    move_data = build_move_data(moves_np, tables.move_name_to_id)
    print(f"  {len(move_data)} moves in lookup")
    status_moves = get_status_moves(move_data)

    # Build type chart
    type_chart = build_type_chart()

    # Generate teams
    print(f"Generating {args.n_teams} teams...")
    pool = np.zeros((args.n_teams, 6, FIELDS_PER_MON), dtype=np.int16)
    generated = 0
    failed = 0

    for i in range(args.n_teams):
        team = random_team(sets_data, base_stats_all, move_data, status_moves,
                           move_lookup, ability_lookup, item_lookup,
                           moves_np, type_chart)
        if team is None:
            failed += 1
            # Retry with fresh shuffle
            team = random_team(sets_data, base_stats_all, move_data, status_moves,
                               move_lookup, ability_lookup, item_lookup,
                               moves_np, type_chart)

        if team is not None:
            for slot, poke_set in enumerate(team):
                pool[i, slot] = set_to_array(poke_set, species_lookup, move_lookup,
                                             ability_lookup, item_lookup,
                                             moves_np, base_stats_all)
            generated += 1
        else:
            failed += 1

        if (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{args.n_teams} (generated={generated}, failed={failed})")

    print(f"Generated {generated}/{args.n_teams} teams ({failed} failures)")

    # Print some stats
    if generated > 0:
        # Species diversity
        species_ids = pool[:generated, :, 0]
        unique_species = len(np.unique(species_ids[species_ids > 0]))
        print(f"  Unique species used: {unique_species}")

        # Item diversity
        item_ids = pool[:generated, :, 2]
        unique_items = len(np.unique(item_ids[item_ids > 0]))
        print(f"  Unique items used: {unique_items}")

        # Move diversity
        move_ids = pool[:generated, :, 12:16]
        unique_moves = len(np.unique(move_ids[move_ids > 0]))
        print(f"  Unique moves used: {unique_moves}")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(args.output, teams=pool[:generated], field_names=FIELD_NAMES)
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"Saved to {args.output} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
