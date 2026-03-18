"""
Showdown TypeScript → NumPy data extractor.

Reads Pokemon Showdown's data/*.ts files directly using regex-based parsing
(no TS compiler needed — the data files are regular object literals).

Output: .npz files saved to pokejax/data/gen{N}/ containing:
  type_chart.npy       float32[N_TYPES, N_TYPES]
  species.npy          structured array or int16[N_SPECIES, SPECIES_FIELDS]
  moves.npy            int16[N_MOVES, MOVE_FIELDS]
  abilities.npy        object array (name → event hooks metadata)
  items.npy            structured or int16[N_ITEMS, ITEM_FIELDS]
  natures.npy          float32[25, 5]
  move_name_to_id.npy  pickled dict
  species_name_to_id.npy
  ability_name_to_id.npy
  item_name_to_id.npy

Usage:
    python -m pokejax.data.extractor --showdown-path /path/to/pokemon-showdown --gen 4
"""

import re
import json
import pickle
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pokejax.types import (
    TYPE_NAMES, N_TYPES,
    STAT_HP, STAT_ATK, STAT_DEF, STAT_SPA, STAT_SPD, STAT_SPE,
)

# ---------------------------------------------------------------------------
# Type chart
# ---------------------------------------------------------------------------

# Row = attacking type, Col = defending type, value = damage multiplier
# Built from known Pokemon type chart (static, doesn't need to parse TS)

def _build_type_chart(gen: int) -> np.ndarray:
    """Return float32[N_TYPES, N_TYPES] type effectiveness table.

    Index 0 is the '???' sentinel type (neutral to everything).
    Indices 1-18 correspond to TYPE_NAMES[1:].
    Gen 4 excludes Fairy (index 18 → all 1.0).
    """
    # Names to index
    t = {name: i for i, name in enumerate(TYPE_NAMES)}

    # Base type chart (Gen 2+, physical/special split era)
    # (attacking_type, defending_type) → multiplier
    super_effective: List[Tuple[str, str]] = [
        # Normal
        # Fire
        ("Fire",     "Grass"), ("Fire",     "Ice"),    ("Fire",     "Bug"),
        ("Fire",     "Steel"),
        # Water
        ("Water",    "Fire"),  ("Water",    "Ground"), ("Water",    "Rock"),
        # Electric
        ("Electric", "Water"), ("Electric", "Flying"),
        # Grass
        ("Grass",    "Water"), ("Grass",    "Ground"), ("Grass",    "Rock"),
        # Ice
        ("Ice",      "Grass"), ("Ice",      "Ground"), ("Ice",      "Flying"),
        ("Ice",      "Dragon"),
        # Fighting
        ("Fighting", "Normal"),("Fighting", "Ice"),    ("Fighting", "Rock"),
        ("Fighting", "Dark"),  ("Fighting", "Steel"),
        # Poison
        ("Poison",   "Grass"), ("Poison",   "Fairy"),
        # Ground
        ("Ground",   "Fire"),  ("Ground",   "Electric"),("Ground",  "Poison"),
        ("Ground",   "Rock"),  ("Ground",   "Steel"),
        # Flying
        ("Flying",   "Grass"), ("Flying",   "Fighting"),("Flying",  "Bug"),
        # Psychic
        ("Psychic",  "Fighting"),("Psychic","Poison"),
        # Bug
        ("Bug",      "Grass"), ("Bug",      "Psychic"), ("Bug",     "Dark"),
        # Rock
        ("Rock",     "Fire"),  ("Rock",     "Ice"),     ("Rock",    "Flying"),
        ("Rock",     "Bug"),
        # Ghost
        ("Ghost",    "Psychic"),("Ghost",   "Ghost"),
        # Dragon
        ("Dragon",   "Dragon"),
        # Dark
        ("Dark",     "Psychic"),("Dark",    "Ghost"),
        # Steel
        ("Steel",    "Ice"),   ("Steel",    "Rock"),    ("Steel",   "Fairy"),
        # Fairy
        ("Fairy",    "Fighting"),("Fairy",  "Dragon"),  ("Fairy",   "Dark"),
    ]

    not_very_effective: List[Tuple[str, str]] = [
        ("Normal",   "Rock"),  ("Normal",  "Steel"),
        ("Fire",     "Fire"),  ("Fire",    "Water"),    ("Fire",    "Rock"),
        ("Fire",     "Dragon"),
        ("Water",    "Water"), ("Water",   "Grass"),    ("Water",   "Dragon"),
        ("Electric", "Electric"),("Electric","Grass"),  ("Electric","Dragon"),
        ("Grass",    "Fire"),  ("Grass",   "Grass"),    ("Grass",   "Poison"),
        ("Grass",    "Flying"),("Grass",   "Bug"),      ("Grass",   "Dragon"),
        ("Grass",    "Steel"),
        ("Ice",      "Fire"),  ("Ice",     "Water"),  ("Ice",     "Ice"),
        ("Ice",      "Steel"),
        ("Fighting", "Poison"),("Fighting","Flying"),  ("Fighting","Psychic"),
        ("Fighting", "Bug"),   ("Fighting","Fairy"),
        ("Poison",   "Poison"),("Poison",  "Ground"),   ("Poison",  "Rock"),
        ("Poison",   "Ghost"),
        ("Ground",   "Grass"), ("Ground",  "Bug"),
        ("Flying",   "Electric"),("Flying","Rock"),     ("Flying",  "Steel"),
        ("Psychic",  "Psychic"),("Psychic","Steel"),
        ("Bug",      "Fire"),  ("Bug",     "Fighting"), ("Bug",     "Poison"),
        ("Bug",      "Flying"),("Bug",     "Ghost"),    ("Bug",     "Steel"),
        ("Bug",      "Fairy"),
        ("Rock",     "Fighting"),("Rock",  "Ground"),   ("Rock",    "Steel"),
        ("Ghost",    "Dark"),  ("Ghost",   "Steel"),
        ("Dragon",   "Steel"),
        ("Dark",     "Fighting"),("Dark",  "Dark"),     ("Dark",    "Steel"),
        ("Dark",     "Fairy"),
        ("Steel",    "Fire"),  ("Steel",   "Water"),    ("Steel",   "Electric"),
        ("Steel",    "Steel"),
        ("Fairy",    "Fire"),  ("Fairy",   "Poison"),   ("Fairy",   "Steel"),
    ]

    immune: List[Tuple[str, str]] = [
        ("Normal",   "Ghost"),
        ("Electric", "Ground"),
        ("Fighting", "Ghost"),
        ("Ground",   "Flying"),
        ("Psychic",  "Dark"),
        ("Ghost",    "Normal"),
        ("Dragon",   "Fairy"),   # Gen 6+ only (skipped below for gen<6)
        ("Poison",   "Steel"),   # Poison-type moves do 0x to Steel in Gen 2-5
    ]

    chart = np.ones((N_TYPES, N_TYPES), dtype=np.float32)
    # Sentinel row/col 0 stays 1.0 (neutral)

    for (atk, dfn) in super_effective:
        if atk not in t or dfn not in t:
            continue
        if gen < 6 and (atk == "Fairy" or dfn == "Fairy"):
            continue
        chart[t[atk], t[dfn]] = 2.0

    for (atk, dfn) in not_very_effective:
        if atk not in t or dfn not in t:
            continue
        if gen < 6 and (atk == "Fairy" or dfn == "Fairy"):
            continue
        chart[t[atk], t[dfn]] = 0.5

    for (atk, dfn) in immune:
        if atk not in t or dfn not in t:
            continue
        if gen < 6 and (atk == "Fairy" or dfn == "Fairy"):
            continue
        chart[t[atk], t[dfn]] = 0.0

    # Ghost/Dark immunity to Psychic was different in Gen 1 (not handled here)
    return chart


# ---------------------------------------------------------------------------
# Natures
# ---------------------------------------------------------------------------

# Nature name → (boosted_stat_idx, hindered_stat_idx)
# Neutral natures boost/hinder the same stat (net 1.0).
# Stat indices: ATK=1, DEF=2, SPA=3, SPD=4, SPE=5 (matching STAT_* without HP)
_NATURE_DATA = {
    # Neutral
    "Hardy":   (1, 1), "Docile":  (2, 2), "Serious": (5, 5),
    "Bashful": (3, 3), "Quirky":  (4, 4),
    # +ATK
    "Lonely":  (1, 2), "Brave":   (1, 5), "Adamant": (1, 3), "Naughty": (1, 4),
    # +DEF
    "Bold":    (2, 1), "Relaxed": (2, 5), "Impish":  (2, 3), "Lax":     (2, 4),
    # +SPE
    "Timid":   (5, 1), "Hasty":   (5, 2), "Jolly":   (5, 3), "Naive":   (5, 4),
    # +SPA
    "Modest":  (3, 1), "Mild":    (3, 2), "Quiet":   (3, 5), "Rash":    (3, 4),
    # +SPD
    "Calm":    (4, 1), "Gentle":  (4, 2), "Sassy":   (4, 5), "Careful": (4, 3),
}
NATURE_NAMES = list(_NATURE_DATA.keys())  # fixed order, index 0-24


def _build_nature_table() -> np.ndarray:
    """Return float32[25, 5] multipliers for [atk, def, spa, spd, spe]."""
    table = np.ones((25, 5), dtype=np.float32)
    for idx, name in enumerate(NATURE_NAMES):
        boosted, hindered = _NATURE_DATA[name]
        boost_col = boosted - 1   # convert STAT index (1-5) to array col (0-4)
        hinder_col = hindered - 1
        if boost_col != hinder_col:
            table[idx, boost_col] = 1.1
            table[idx, hinder_col] = 0.9
    return table


# ---------------------------------------------------------------------------
# TypeScript object-literal parser
# ---------------------------------------------------------------------------

def _strip_ts_comments(text: str) -> str:
    """Remove // line comments and /* block comments */ from TS source."""
    # Block comments
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    # Line comments
    text = re.sub(r'//[^\n]*', '', text)
    return text


def _extract_object(text: str, var_name: str) -> Optional[str]:
    """
    Find `export const <var_name>: ... = { ... };` and return the inner JSON-like block.
    Returns the raw brace content (not parsed).
    """
    pattern = rf'export\s+const\s+{re.escape(var_name)}\s*[^=]*=\s*(\{{)'
    m = re.search(pattern, text)
    if not m:
        return None
    start = m.start(1)
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None


def _ts_obj_to_entries(raw: str) -> List[Tuple[str, str]]:
    """
    Parse the top-level keys from a TS object literal.
    Returns list of (key, raw_value_string) pairs.
    This is a best-effort parser; handles the regular structure of Showdown's data files.
    """
    # Strip outer braces
    inner = raw.strip()[1:-1]
    entries = []
    i = 0
    n = len(inner)
    while i < n:
        # Skip whitespace / commas
        while i < n and inner[i] in ' \t\n\r,':
            i += 1
        if i >= n:
            break

        # Read key (identifier or quoted string)
        if inner[i] == '"' or inner[i] == "'":
            quote = inner[i]
            j = inner.index(quote, i + 1)
            key = inner[i+1:j]
            i = j + 1
        else:
            j = i
            while j < n and inner[j] not in ': \t\n\r':
                j += 1
            key = inner[i:j]
            i = j

        # Skip whitespace and colon
        while i < n and inner[i] in ' \t\n\r:':
            i += 1

        if i >= n:
            break

        # Read value (either an object or a scalar)
        val_start = i
        if inner[i] == '{':
            depth = 0
            while i < n:
                if inner[i] == '{':
                    depth += 1
                elif inner[i] == '}':
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
        elif inner[i] == '[':
            depth = 0
            while i < n:
                if inner[i] == '[':
                    depth += 1
                elif inner[i] == ']':
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
        else:
            # scalar: read until comma or end
            while i < n and inner[i] not in ',\n':
                i += 1
        val = inner[val_start:i].strip().rstrip(',')
        if key:
            entries.append((key, val))
    return entries


def _parse_value(val: str) -> Any:
    """Parse a single value string into a Python scalar, list, or dict."""
    val = val.strip()
    if val in ('true', 'True'):
        return True
    elif val in ('false', 'False'):
        return False
    elif val == 'null':
        return None
    elif val.startswith('"') or val.startswith("'"):
        # Quoted string — strip quotes
        return val[1:-1] if len(val) >= 2 else val
    elif val.startswith('{'):
        # Nested object — recurse
        return _parse_inner_dict(val)
    elif val.startswith('['):
        # Array — parse elements
        inner = val[1:-1].strip()
        if not inner:
            return []
        items = []
        depth = 0
        current = []
        for ch in inner:
            if ch in ('{', '['):
                depth += 1
            elif ch in ('}', ']'):
                depth -= 1
            if ch == ',' and depth == 0:
                items.append(''.join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            items.append(''.join(current).strip())
        return [_parse_value(item) for item in items if item]
    else:
        try:
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                return val  # raw string


def _parse_inner_dict(raw: str) -> Dict[str, Any]:
    """Best-effort parse of a TS object literal's fields into a Python dict."""
    result: Dict[str, Any] = {}
    for key, val in _ts_obj_to_entries(raw):
        result[key] = _parse_value(val.strip())
    return result


# ---------------------------------------------------------------------------
# Species extraction
# ---------------------------------------------------------------------------

SPECIES_FIELDS = 12  # hp, atk, def, spa, spd, spe, type1, type2, weight_hg, ability1, ability2, ability_h

def _extract_species(showdown_path: Path, gen: int,
                     type_name_to_id: Dict[str, int],
                     ) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
    """
    Parse data/pokedex.ts and any gen-specific overrides.
    Returns:
        data: int16[N_SPECIES, SPECIES_FIELDS]
        name_to_id: dict mapping species name → row index
        names: list of species names (index → name)
    """
    pokedex_file = showdown_path / "data" / "pokedex.ts"
    if not pokedex_file.exists():
        raise FileNotFoundError(f"Not found: {pokedex_file}")

    text = _strip_ts_comments(pokedex_file.read_text(encoding="utf-8"))
    raw_block = _extract_object(text, "Pokedex")
    if raw_block is None:
        raise ValueError("Could not find 'Pokedex' object in pokedex.ts")

    entries = _ts_obj_to_entries(raw_block)
    records: List[Dict[str, Any]] = []
    names: List[str] = []

    for key, val in entries:
        d = _parse_inner_dict(val)
        if not d:
            continue
        name = d.get("name", key)
        records.append(d)
        names.append(str(name))

    n = len(records)
    data = np.zeros((n, SPECIES_FIELDS), dtype=np.int16)

    for i, d in enumerate(records):
        bs = d.get("baseStats", {})
        if isinstance(bs, dict):
            data[i, 0] = bs.get("hp",  45)
            data[i, 1] = bs.get("atk", 45)
            data[i, 2] = bs.get("def", 45)
            data[i, 3] = bs.get("spa", 45)
            data[i, 4] = bs.get("spd", 45)
            data[i, 5] = bs.get("spe", 45)

        types_raw = d.get("types", [])
        if isinstance(types_raw, list) and len(types_raw) >= 1:
            data[i, 6] = type_name_to_id.get(str(types_raw[0]), 1)
            data[i, 7] = type_name_to_id.get(str(types_raw[1]) if len(types_raw) > 1 else types_raw[0], 0)
        else:
            data[i, 6] = 1  # Normal
            data[i, 7] = 0  # None

        weight_raw = d.get("weightkg", 10)
        try:
            data[i, 8] = int(float(str(weight_raw)) * 10)
        except (ValueError, TypeError):
            data[i, 8] = 100

        # Abilities stored as string IDs — we'll resolve them after ability table is built
        # Placeholder: store 0
        data[i, 9] = 0
        data[i, 10] = 0
        data[i, 11] = 0

    name_to_id = {name: i for i, name in enumerate(names)}
    return data, name_to_id, names


# ---------------------------------------------------------------------------
# Move extraction
# ---------------------------------------------------------------------------

MOVE_FIELDS = 22
# Fields: [base_power, accuracy, type, category, priority, pp,
#          target, flags_lo, flags_hi, crit_ratio,
#          secondary_chance, secondary_status, secondary_boost_stat, secondary_boost_amount,
#          drain_num, drain_den, recoil_num, recoil_den,
#          multi_hit_min, multi_hit_max,
#          heal_num, heal_den]

_CATEGORY_MAP = {"Physical": 0, "Special": 1, "Status": 2}
_TARGET_MAP = {
    "normal": 0, "self": 1, "adjacentAlly": 2, "adjacentAllyOrSelf": 3,
    "adjacentFoe": 4, "allAdjacentFoes": 5, "allAdjacent": 6,
    "any": 7, "all": 8, "allyTeam": 9, "foeSide": 10, "allySide": 11,
    "allies": 12, "randomNormal": 13, "scripted": 14,
}

def _extract_moves(showdown_path: Path, gen: int,
                   type_name_to_id: Dict[str, int],
                   ) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
    moves_file = showdown_path / "data" / "moves.ts"
    if not moves_file.exists():
        raise FileNotFoundError(f"Not found: {moves_file}")

    text = _strip_ts_comments(moves_file.read_text(encoding="utf-8"))
    raw_block = _extract_object(text, "Moves")
    if raw_block is None:
        raise ValueError("Could not find 'Moves' in moves.ts")

    entries = _ts_obj_to_entries(raw_block)
    records: List[Dict[str, Any]] = []
    names: List[str] = []

    for key, val in entries:
        d = _parse_inner_dict(val)
        if not d:
            continue
        name = d.get("name", key)
        records.append(d)
        names.append(str(name))

    n = len(records)
    data = np.zeros((n, MOVE_FIELDS), dtype=np.int16)

    for i, d in enumerate(records):
        bp = d.get("basePower", 0)
        data[i, 0] = int(bp) if isinstance(bp, (int, float)) else 0

        acc = d.get("accuracy", 100)
        # 'true' accuracy means always hits — store as 101 (sentinel)
        if acc is True or acc == "true":
            data[i, 1] = 101
        else:
            try:
                data[i, 1] = int(acc)
            except (ValueError, TypeError):
                data[i, 1] = 100

        move_type = d.get("type", "Normal")
        data[i, 2] = type_name_to_id.get(str(move_type), 1)

        category = d.get("category", "Status")
        data[i, 3] = _CATEGORY_MAP.get(str(category), 2)

        try:
            data[i, 4] = int(d.get("priority", 0))
        except (ValueError, TypeError):
            data[i, 4] = 0
        try:
            data[i, 5] = int(d.get("pp", 10))
        except (ValueError, TypeError):
            data[i, 5] = 10

        target = d.get("target", "normal")
        data[i, 6] = _TARGET_MAP.get(str(target), 0)

        # flags — parse from flags dict {contact: 1, protect: 1, punch: 1, ...}
        flags_raw = d.get("flags", {})
        if not isinstance(flags_raw, dict):
            flags_raw = {}
        flag_bits = 0
        if flags_raw.get("contact"):  flag_bits |= (1 << 0)
        if flags_raw.get("protect"):  flag_bits |= (1 << 1)
        if flags_raw.get("mirror"):   flag_bits |= (1 << 2)
        if flags_raw.get("sound"):    flag_bits |= (1 << 3)
        if flags_raw.get("punch"):    flag_bits |= (1 << 4)
        if flags_raw.get("bite"):     flag_bits |= (1 << 5)
        if flags_raw.get("bullet"):   flag_bits |= (1 << 6)
        if flags_raw.get("defrost"):  flag_bits |= (1 << 7)
        if flags_raw.get("powder"):   flag_bits |= (1 << 8)
        if flags_raw.get("snatch"):   flag_bits |= (1 << 9)
        if flags_raw.get("heal"):     flag_bits |= (1 << 10)
        if flags_raw.get("recharge"): flag_bits |= (1 << 11)
        data[i, 7] = flag_bits & 0xFF           # lower 8 bits
        data[i, 8] = (flag_bits >> 8) & 0xFF    # upper 8 bits

        data[i, 9] = int(d.get("critRatio", 1))

        # Secondary effects (from `secondary: { chance, status, boosts }`)
        status_map = {"brn": 1, "psn": 2, "tox": 3, "slp": 4, "frz": 5, "par": 6}
        boost_stat_map = {
            "atk": 0, "def": 1, "spa": 2, "spd": 3, "spe": 4,
            "accuracy": 5, "evasion": 6,
        }
        sec = d.get("secondary", None)
        if sec and isinstance(sec, dict):
            data[i, 10] = int(sec.get("chance", 0))
            sec_status = sec.get("status", "")
            data[i, 11] = status_map.get(str(sec_status), 0)
            # Secondary stat change targeting foe (e.g. Psychic -SPD, Crunch -DEF)
            # Moves with `self` inside secondary (Charge Beam self-boost) are handled
            # via the move_effects table instead.
            sec_boosts = sec.get("boosts", {})
            if sec_boosts and isinstance(sec_boosts, dict):
                for stat_name, amt in sec_boosts.items():
                    stat_idx = boost_stat_map.get(str(stat_name), -1)
                    if stat_idx >= 0 and isinstance(amt, (int, float)) and int(amt) != 0:
                        data[i, 12] = stat_idx
                        data[i, 13] = int(amt)
                        break   # store first boost only
        else:
            data[i, 10] = 0
            data[i, 11] = 0

        # Primary status infliction (move-level `status` field, e.g. Will-O-Wisp)
        # Encode as 100 % secondary so the existing hit-pipeline applies it.
        if data[i, 10] == 0 and data[i, 11] == 0:
            primary_status = d.get("status", "")
            ps_val = status_map.get(str(primary_status), 0)
            if ps_val > 0:
                data[i, 10] = 100   # 100 % chance = primary effect
                data[i, 11] = ps_val

        # Drain
        drain = d.get("drain", None)
        if drain and isinstance(drain, list) and len(drain) == 2:
            data[i, 14] = int(drain[0])
            data[i, 15] = int(drain[1])

        # Recoil
        recoil = d.get("recoil", None)
        if recoil and isinstance(recoil, list) and len(recoil) == 2:
            data[i, 16] = int(recoil[0])
            data[i, 17] = int(recoil[1])

        # Multi-hit
        mh = d.get("multihit", None)
        if mh is not None:
            if isinstance(mh, list) and len(mh) == 2:
                data[i, 18] = int(mh[0])
                data[i, 19] = int(mh[1])
            elif isinstance(mh, (int, float)):
                data[i, 18] = int(mh)
                data[i, 19] = int(mh)

        # Heal
        heal = d.get("heal", None)
        if heal and isinstance(heal, list) and len(heal) == 2:
            data[i, 20] = int(heal[0])
            data[i, 21] = int(heal[1])

    name_to_id = {name: i for i, name in enumerate(names)}
    return data, name_to_id, names


# ---------------------------------------------------------------------------
# Ability extraction
# ---------------------------------------------------------------------------

def _extract_abilities(showdown_path: Path, gen: int,
                       ) -> Tuple[Dict[str, int], List[str]]:
    """
    Parse data/abilities.ts and return (name_to_id, names).

    We only need ability names and a sequential ID mapping — the actual
    event implementations are hand-coded in mechanics/abilities.py.
    """
    ability_file = showdown_path / "data" / "abilities.ts"
    if not ability_file.exists():
        raise FileNotFoundError(f"Not found: {ability_file}")

    text = _strip_ts_comments(ability_file.read_text(encoding="utf-8"))
    raw_block = _extract_object(text, "Abilities")
    if raw_block is None:
        raise ValueError("Could not find 'Abilities' in abilities.ts")

    entries = _ts_obj_to_entries(raw_block)
    names: List[str] = []
    for key, val in entries:
        d = _parse_inner_dict(val)
        if not d:
            continue
        name = d.get("name", key)
        names.append(str(name))

    name_to_id = {name: i for i, name in enumerate(names)}
    return name_to_id, names


# ---------------------------------------------------------------------------
# Item extraction
# ---------------------------------------------------------------------------

ITEM_FIELDS = 4
# [item_type, fling_power, fling_status, natural_gift_type]
# Most item behavior is hardcoded in mechanics/items.py; only metadata here.

def _extract_items(showdown_path: Path, gen: int,
                   ) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
    """
    Parse data/items.ts and return (data, name_to_id, names).
    """
    items_file = showdown_path / "data" / "items.ts"
    if not items_file.exists():
        raise FileNotFoundError(f"Not found: {items_file}")

    text = _strip_ts_comments(items_file.read_text(encoding="utf-8"))
    raw_block = _extract_object(text, "Items")
    if raw_block is None:
        raise ValueError("Could not find 'Items' in items.ts")

    entries = _ts_obj_to_entries(raw_block)
    records: List[Dict[str, Any]] = []
    names: List[str] = []

    for key, val in entries:
        d = _parse_inner_dict(val)
        if not d:
            continue
        name = d.get("name", key)
        records.append(d)
        names.append(str(name))

    n = len(records)
    data = np.zeros((n, ITEM_FIELDS), dtype=np.int16)
    for i, d in enumerate(records):
        data[i, 0] = 1 if d.get("isBerry") else 0
        data[i, 1] = int(d.get("fling", {}).get("basePower", 0)
                         if isinstance(d.get("fling"), dict) else 0)

    name_to_id = {name: i for i, name in enumerate(names)}
    return data, name_to_id, names


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def extract(showdown_path: Path, gen: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Type index
    type_name_to_id = {name: i for i, name in enumerate(TYPE_NAMES)}

    print("Building type chart...")
    type_chart = _build_type_chart(gen)
    np.save(output_dir / "type_chart.npy", type_chart)
    print(f"  Saved type_chart {type_chart.shape}")

    print("Building nature table...")
    nature_table = _build_nature_table()
    np.save(output_dir / "natures.npy", nature_table)
    print(f"  Saved natures {nature_table.shape}")

    print("Extracting species...")
    try:
        species_data, species_name_to_id, species_names = _extract_species(
            showdown_path, gen, type_name_to_id
        )
        np.save(output_dir / "species.npy", species_data)
        with open(output_dir / "species_name_to_id.pkl", "wb") as f:
            pickle.dump(species_name_to_id, f)
        with open(output_dir / "species_names.pkl", "wb") as f:
            pickle.dump(species_names, f)
        print(f"  Saved species {species_data.shape} ({len(species_names)} species)")
    except Exception as e:
        print(f"  WARNING: species extraction failed: {e}")

    print("Extracting moves...")
    try:
        move_data, move_name_to_id, move_names = _extract_moves(
            showdown_path, gen, type_name_to_id
        )
        np.save(output_dir / "moves.npy", move_data)
        with open(output_dir / "move_name_to_id.pkl", "wb") as f:
            pickle.dump(move_name_to_id, f)
        with open(output_dir / "move_names.pkl", "wb") as f:
            pickle.dump(move_names, f)
        print(f"  Saved moves {move_data.shape} ({len(move_names)} moves)")
    except Exception as e:
        print(f"  WARNING: move extraction failed: {e}")

    print("Extracting abilities...")
    try:
        ability_name_to_id, ability_names = _extract_abilities(showdown_path, gen)
        with open(output_dir / "ability_name_to_id.pkl", "wb") as f:
            pickle.dump(ability_name_to_id, f)
        with open(output_dir / "ability_names.pkl", "wb") as f:
            pickle.dump(ability_names, f)
        print(f"  Saved abilities ({len(ability_names)} abilities)")
    except Exception as e:
        print(f"  WARNING: ability extraction failed: {e}")

    print("Extracting items...")
    try:
        item_data, item_name_to_id, item_names = _extract_items(showdown_path, gen)
        np.save(output_dir / "items.npy", item_data)
        with open(output_dir / "item_name_to_id.pkl", "wb") as f:
            pickle.dump(item_name_to_id, f)
        with open(output_dir / "item_names.pkl", "wb") as f:
            pickle.dump(item_names, f)
        print(f"  Saved items {item_data.shape} ({len(item_names)} items)")
    except Exception as e:
        print(f"  WARNING: item extraction failed: {e}")

    # Save metadata
    meta = {"gen": gen, "n_types": N_TYPES, "nature_names": NATURE_NAMES}
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nExtraction complete → {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Showdown data to NumPy arrays")
    parser.add_argument("--showdown-path", required=True,
                        help="Path to pokemon-showdown repo root")
    parser.add_argument("--gen", type=int, default=4,
                        help="Generation to extract (default: 4)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: pokejax/data/gen{N}/)")
    args = parser.parse_args()

    showdown_path = Path(args.showdown_path)
    if not showdown_path.exists():
        raise SystemExit(f"Showdown path not found: {showdown_path}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / f"gen{args.gen}"

    extract(showdown_path, args.gen, output_dir)


if __name__ == "__main__":
    main()
