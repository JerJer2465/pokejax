"""
poke-env Player for the pokejax PokeTransformer model.

Converts poke-env's Battle object into pokejax's obs format
(int_ids, float_feats, legal_mask), runs the model, and maps
the predicted action back to a poke-env BattleOrder.

Usage:
    from pokejax.players.showdown_player import PokejaxPlayer
    player = PokejaxPlayer(checkpoint_path="checkpoints/bc_final.pkl")
    await player.battle_against(opponent, n_battles=10)
"""

from __future__ import annotations

import os
import re
import pickle
from typing import Optional, Dict, List

import numpy as np
import jax
import jax.numpy as jnp

from poke_env.player import Player
from poke_env.environment import (
    AbstractBattle,
    Pokemon,
    Move,
    Weather,
    Field,
    SideCondition,
    PokemonType,
    Status,
)

try:
    from poke_env.environment import Effect
except ImportError:
    Effect = None

from pokejax.rl.model import PokeTransformer
from pokejax.data.tables import load_tables


# ---------------------------------------------------------------------------
# Name normalization (match pokejax tables)
# ---------------------------------------------------------------------------

def _normalize(name: str) -> str:
    """Normalize a name for pokejax table lookup."""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


# ---------------------------------------------------------------------------
# poke-env enum → string mappings
# ---------------------------------------------------------------------------

_STATUS_MAP = {
    Status.BRN: 1, Status.PSN: 2, Status.TOX: 3,
    Status.SLP: 4, Status.FRZ: 5, Status.PAR: 6,
    None: 0,
}

_TYPE_MAP = {
    PokemonType.NORMAL: 1, PokemonType.FIRE: 2, PokemonType.WATER: 3,
    PokemonType.ELECTRIC: 4, PokemonType.GRASS: 5, PokemonType.ICE: 6,
    PokemonType.FIGHTING: 7, PokemonType.POISON: 8, PokemonType.GROUND: 9,
    PokemonType.FLYING: 10, PokemonType.PSYCHIC: 11, PokemonType.BUG: 12,
    PokemonType.ROCK: 13, PokemonType.GHOST: 14, PokemonType.DRAGON: 15,
    PokemonType.DARK: 16, PokemonType.STEEL: 17, PokemonType.FAIRY: 18,
}

_WEATHER_MAP = {
    Weather.SUNNYDAY: 1, Weather.RAINDANCE: 2,
    Weather.SANDSTORM: 3, Weather.HAIL: 4,
}
# Add Gen8+ variants safely
for _attr, _val in [("SNOW", 4), ("DESOLATELAND", 1),
                    ("PRIMORDIALSEA", 2), ("DELTASTREAM", 0)]:
    try:
        _WEATHER_MAP[getattr(Weather, _attr)] = _val
    except AttributeError:
        pass

# Volatile status mapping for float features
_VOLATILE_NAMES = {
    "confusion": 0, "infatuation": 1, "leechseed": 2, "curse": 3,
    "aquaring": 4, "ingrain": 5, "taunt": 6, "encore": 7,
    "flinch": 8, "embargo": 9, "healblock": 10, "magnetrise": 11,
    "partiallytrapped": 12, "perishsong": 13, "powertrick": 14,
    "substitute": 15, "yawn": 16, "focusenergy": 17, "charge": 18,
    "stockpile": 19, "torment": 20, "nightmare": 21, "imprison": 22,
    "mustrecharge": 23, "twoturnmove": 24, "destinybond": 25, "grudge": 26,
}

_N_VOLATILE = 27

# Effect enum → volatile name
_EFFECT_TO_VOLATILE = {}
if Effect is not None:
    _EFF_MAP = {
        "CONFUSION": "confusion", "ATTRACT": "infatuation",
        "LEECH_SEED": "leechseed", "CURSE": "curse",
        "AQUA_RING": "aquaring", "INGRAIN": "ingrain",
        "TAUNT": "taunt", "ENCORE": "encore",
        "FLINCH": "flinch", "EMBARGO": "embargo",
        "FOCUS_ENERGY": "focusenergy", "SUBSTITUTE": "substitute",
        "YAWN": "yawn", "TORMENT": "torment", "NIGHTMARE": "nightmare",
        "DESTINY_BOND": "destinybond",
    }
    for attr, val in _EFF_MAP.items():
        try:
            _EFFECT_TO_VOLATILE[getattr(Effect, attr)] = val
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Observation builder constants (match pokejax/rl/obs_builder.py)
# ---------------------------------------------------------------------------

FLOAT_DIM = 394
FIELD_DIM = 84
N_TOKENS = 15
N_ACTIONS = 10
INT_IDS_PER_TOKEN = 8
_N_PS_TYPES = 18

# Float feature offsets (per-pokemon)
_OFF_HP_FRAC    = 0
_OFF_HP_BIN     = 1    # 10 dims
_OFF_BASE_STATS = 11   # 6 dims
_OFF_BOOSTS     = 17   # 91 dims (7 stats × 13 levels)
_OFF_STATUS     = 108  # 7 dims
_OFF_VOLATILE   = 115  # 27 dims
_OFF_TYPE1      = 142  # 18 dims
_OFF_TYPE2      = 160  # 18 dims
_OFF_IS_FAINTED = 178
_OFF_IS_ACTIVE  = 179
_OFF_SLOT       = 180  # 6 dims
_OFF_IS_OWN     = 186
_OFF_MOVES      = 187  # 4×45 = 180 dims
_OFF_SLEEP_BIN  = 367  # 4 dims
_OFF_REST_BIN   = 371  # 3 dims
_OFF_SUB_FRAC   = 374
_OFF_FORCE_TRAP = 375
_OFF_MOV_DIS    = 376  # 4 dims
_OFF_CONF_BIN   = 380  # 4 dims
_OFF_TAUNT      = 384
_OFF_ENCORE     = 385
_OFF_YAWN       = 386
_OFF_LEVEL      = 387
_OFF_PERISH_BIN = 388  # 4 dims
_OFF_PROTECT    = 392
_OFF_LOCKED_MOV = 393

# Move sub-offsets (within 45-dim block)
_MOFF_BP_BIN  = 0   # 8 dims
_MOFF_ACC_BIN = 8   # 6 dims
_MOFF_TYPE    = 14  # 18 dims
_MOFF_CAT     = 32  # 3 dims
_MOFF_PRI     = 35  # 8 dims
_MOFF_PP      = 43  # 1 dim
_MOFF_KNOWN   = 44  # 1 dim

# Field offsets
_FOFF_WEATHER     = 0   # 5 dims
_FOFF_WT_TURNS    = 5   # 8 dims
_FOFF_PSEUDO      = 13  # 5 dims
_FOFF_TR_TURNS    = 18  # 4 dims
_FOFF_HAZARDS_OWN = 22  # 7 dims
_FOFF_HAZARDS_OPP = 29  # 7 dims
_FOFF_SCREENS_OWN = 36  # 6 dims
_FOFF_SCREENS_OPP = 42  # 6 dims
_FOFF_TURN_BIN    = 48  # 10 dims
_FOFF_FAINTED     = 58  # 2 dims
_FOFF_TOXIC_OWN   = 60  # 5 dims
_FOFF_TOXIC_OPP   = 65  # 5 dims
_FOFF_TAILWIND    = 70  # 2 dims
_FOFF_WISH        = 72  # 2 dims
_FOFF_SAFEGUARD   = 74  # 2 dims
_FOFF_MIST        = 76  # 2 dims
_FOFF_LUCKY_CHANT = 78  # 2 dims
_FOFF_GRAVITY_T   = 80  # 4 dims

# Binning thresholds
_BP_THRESHOLDS  = np.array([0, 1, 41, 61, 81, 101, 121, 151])
_ACC_THRESHOLDS = np.array([0, 50, 70, 80, 90, 100])
_HP_THRESHOLDS  = np.array([0, 10, 20, 33, 50, 66, 75, 88, 100])
_TURN_THRESHOLDS = np.array([1, 2, 4, 6, 9, 13, 18, 25, 35])


def _bin_idx(val, thresholds):
    idx = np.sum(val >= thresholds) - 1
    return max(0, min(idx, len(thresholds) - 1))


def _bin_onehot(val, thresholds):
    n = len(thresholds)
    oh = np.zeros(n, dtype=np.float32)
    oh[_bin_idx(val, thresholds)] = 1.0
    return oh


def _onehot(idx, n):
    oh = np.zeros(n, dtype=np.float32)
    if 0 <= idx < n:
        oh[idx] = 1.0
    return oh


# ---------------------------------------------------------------------------
# Battle → obs conversion
# ---------------------------------------------------------------------------

class ObsBridge:
    """Converts poke-env Battle objects to pokejax model inputs."""

    def __init__(self, tables):
        self.tables = tables
        # Build normalized name → id lookups
        self._species_lookup = self._build_lookup(tables.species_name_to_id)
        self._move_lookup = self._build_lookup(tables.move_name_to_id)
        self._ability_lookup = self._build_lookup(tables.ability_name_to_id)
        self._item_lookup = self._build_lookup(tables.item_name_to_id)
        # Cache numpy move data
        self._moves_np = np.array(tables.moves)

    def _build_lookup(self, name_to_id: dict) -> dict:
        lookup = {}
        for k, v in name_to_id.items():
            lookup[k] = v
            lookup[_normalize(k)] = v
        return lookup

    def _find_id(self, name: str, lookup: dict) -> int:
        if name in lookup:
            return lookup[name]
        norm = _normalize(name)
        if norm in lookup:
            return lookup[norm]
        return 0

    def _species_id(self, pokemon: Pokemon) -> int:
        return self._find_id(pokemon.species, self._species_lookup)

    def _move_id(self, move: Move) -> int:
        return self._find_id(move.id, self._move_lookup)

    def _ability_id(self, pokemon: Pokemon) -> int:
        if pokemon.ability:
            return self._find_id(pokemon.ability, self._ability_lookup)
        return 0

    def _item_id(self, pokemon: Pokemon) -> int:
        if pokemon.item:
            return self._find_id(pokemon.item, self._item_lookup)
        return 0

    def _encode_move_feats(self, move: Optional[Move], is_known: bool = True) -> tuple:
        """Returns (move_int_id, 45-dim float features)."""
        feats = np.zeros(45, dtype=np.float32)

        if move is None or not is_known:
            feats[_MOFF_PP] = 1.0  # pp_frac = 1 for unknown
            return 0, feats

        mid = self._move_id(move)

        # Get move data from tables
        if 0 < mid < len(self._moves_np):
            row = self._moves_np[mid]
            bp = int(row[0])
            acc = int(row[1])
            type_id = int(row[2])
            category = int(row[3])
            priority = int(row[4])
        else:
            bp = getattr(move, 'base_power', 0) or 0
            acc = getattr(move, 'accuracy', 100) or 100
            type_id = _TYPE_MAP.get(getattr(move, 'type', None), 1)
            category = 2 if bp == 0 else 0  # rough guess
            priority = getattr(move, 'priority', 0) or 0

        # PS type index (0-based)
        ps_type = max(0, min(type_id - 1, _N_PS_TYPES - 1))

        feats[_MOFF_BP_BIN:_MOFF_BP_BIN + 8] = _bin_onehot(bp, _BP_THRESHOLDS)
        acc_clamped = 0 if acc > 100 else acc
        feats[_MOFF_ACC_BIN:_MOFF_ACC_BIN + 6] = _bin_onehot(acc_clamped, _ACC_THRESHOLDS)
        feats[_MOFF_TYPE:_MOFF_TYPE + 18] = _onehot(ps_type, 18)
        feats[_MOFF_CAT:_MOFF_CAT + 3] = _onehot(category, 3)
        pri_idx = max(0, min(priority + 3, 7))
        feats[_MOFF_PRI:_MOFF_PRI + 8] = _onehot(pri_idx, 8)

        pp = move.current_pp if move.current_pp is not None else 1
        max_pp = move.max_pp if move.max_pp else max(pp, 1)
        pp_frac = pp / max(max_pp, 1)
        feats[_MOFF_PP] = pp_frac
        feats[_MOFF_KNOWN] = 1.0

        return mid, feats

    def _encode_pokemon(self, pokemon: Pokemon, is_own: bool, is_active: bool,
                        slot: int, available_moves: Optional[List[Move]] = None) -> tuple:
        """Returns (int_ids[8], float_feats[394])."""
        int_ids = np.zeros(INT_IDS_PER_TOKEN, dtype=np.int32)
        buf = np.zeros(FLOAT_DIM, dtype=np.float32)

        species_id = self._species_id(pokemon)
        ability_id = self._ability_id(pokemon) if is_own or pokemon.ability else 0
        item_id = self._item_id(pokemon) if is_own or pokemon.item else 0

        # Moves — for active own mon, use available_moves for correct ordering
        if is_active and is_own and available_moves is not None:
            moves_ordered = list(available_moves)
            while len(moves_ordered) < 4:
                moves_ordered.append(None)
            moves_ordered = moves_ordered[:4]
        else:
            pm = list(pokemon.moves.values()) if pokemon.moves else []
            moves_ordered = pm[:4]
            while len(moves_ordered) < 4:
                moves_ordered.append(None)

        move_int_ids = []
        move_feats = []
        for m in moves_ordered:
            mid, mf = self._encode_move_feats(m, is_known=(is_own or m is not None))
            move_int_ids.append(mid)
            move_feats.append(mf)

        # Last move (approximation — poke-env doesn't always track this perfectly)
        last_move_id = 0
        if hasattr(pokemon, 'last_move') and pokemon.last_move:
            last_move_id = self._move_id(pokemon.last_move)

        int_ids[0] = species_id
        int_ids[1] = move_int_ids[0]
        int_ids[2] = move_int_ids[1]
        int_ids[3] = move_int_ids[2]
        int_ids[4] = move_int_ids[3]
        int_ids[5] = ability_id
        int_ids[6] = item_id
        int_ids[7] = last_move_id

        # --- Float features ---
        # HP
        if is_own:
            hp = pokemon.current_hp or 0
            maxhp = pokemon.max_hp or max(hp, 1)
        else:
            bst = getattr(pokemon, 'base_stats', {}) or {}
            base_hp = bst.get('hp', 80)
            maxhp = 2 * base_hp + 141  # rough estimate
            hp = int(pokemon.current_hp_fraction * maxhp)

        hp_frac = hp / max(maxhp, 1)
        buf[_OFF_HP_FRAC] = hp_frac
        hp_pct = max(0, min(int(hp_frac * 100), 100))
        buf[_OFF_HP_BIN:_OFF_HP_BIN + 10] = 0
        buf[_OFF_HP_BIN + _bin_idx(hp_pct, _HP_THRESHOLDS)] = 1.0

        # Base stats (normalized /255)
        bst = getattr(pokemon, 'base_stats', {}) or {}
        stats = [bst.get('hp', 80), bst.get('atk', 80), bst.get('def', 80),
                 bst.get('spa', 80), bst.get('spd', 80), bst.get('spe', 80)]
        for i, s in enumerate(stats):
            buf[_OFF_BASE_STATS + i] = s / 255.0

        # Boosts (7 stats × 13 levels)
        boosts = dict(pokemon.boosts) if pokemon.boosts else {}
        boost_order = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion']
        for i, bname in enumerate(boost_order):
            b = boosts.get(bname, 0)
            b_idx = max(0, min(b + 6, 12))
            buf[_OFF_BOOSTS + i * 13 + b_idx] = 1.0

        # Status (7-dim one-hot)
        status_code = _STATUS_MAP.get(pokemon.status, 0)
        buf[_OFF_STATUS + max(0, min(status_code, 6))] = 1.0

        # Volatile statuses (27-dim multi-hot)
        effects = getattr(pokemon, 'effects', {}) or {}
        for eff, _ in effects.items():
            vol_name = _EFFECT_TO_VOLATILE.get(eff)
            if vol_name and vol_name in _VOLATILE_NAMES:
                vol_idx = _VOLATILE_NAMES[vol_name]
                buf[_OFF_VOLATILE + vol_idx] = 1.0

        # Types
        types_raw = pokemon.types if pokemon.types else (PokemonType.NORMAL,)
        t1 = _TYPE_MAP.get(types_raw[0], 1) if types_raw[0] else 1
        ps_t1 = max(0, min(t1 - 1, _N_PS_TYPES - 1))
        buf[_OFF_TYPE1 + ps_t1] = 1.0
        if len(types_raw) > 1 and types_raw[1] is not None:
            t2 = _TYPE_MAP.get(types_raw[1], 0)
            if t2 > 0:
                ps_t2 = max(0, min(t2 - 1, _N_PS_TYPES - 1))
                buf[_OFF_TYPE2 + ps_t2] = 1.0

        # Fainted, active, slot, is_own
        buf[_OFF_IS_FAINTED] = 1.0 if pokemon.fainted else 0.0
        buf[_OFF_IS_ACTIVE] = 1.0 if is_active else 0.0
        if 0 <= slot < 6:
            buf[_OFF_SLOT + slot] = 1.0
        buf[_OFF_IS_OWN] = 1.0 if is_own else 0.0

        # Move features (4 × 45)
        for m_idx in range(4):
            moff = _OFF_MOVES + m_idx * 45
            buf[moff:moff + 45] = move_feats[m_idx]

        # Sleep turns (rough estimate)
        if status_code == 4:  # SLP
            sleep_t = min(getattr(pokemon, 'sleep_turns', 1) or 1, 3)
            buf[_OFF_SLEEP_BIN + sleep_t] = 1.0
        else:
            buf[_OFF_SLEEP_BIN] = 1.0

        # Rest turns bin
        buf[_OFF_REST_BIN] = 1.0  # default: 0 rest turns

        # Level
        level = pokemon.level if pokemon.level else 100
        buf[_OFF_LEVEL] = level / 100.0

        # Confusion, taunt, encore, yawn from effects
        for eff, _ in effects.items():
            vol_name = _EFFECT_TO_VOLATILE.get(eff)
            if vol_name == "confusion":
                buf[_OFF_CONF_BIN + 1] = 1.0  # assume 1 turn
            elif vol_name == "taunt":
                buf[_OFF_TAUNT] = 1.0
            elif vol_name == "encore":
                buf[_OFF_ENCORE] = 1.0
            elif vol_name == "yawn":
                buf[_OFF_YAWN] = 1.0

        # Perish bin (default: 0)
        buf[_OFF_PERISH_BIN] = 1.0

        return int_ids, buf

    def _encode_field(self, battle: AbstractBattle) -> np.ndarray:
        """Encode field token (84 dims)."""
        buf = np.zeros(FIELD_DIM, dtype=np.float32)

        # Weather
        weather_code = 0
        weather_turns = 0
        for w, turns in (battle.weather or {}).items():
            weather_code = _WEATHER_MAP.get(w, 0)
            weather_turns = turns
            break
        buf[_FOFF_WEATHER + max(0, min(weather_code, 4))] = 1.0
        buf[_FOFF_WT_TURNS + max(0, min(weather_turns, 7))] = 1.0

        # Trick room / gravity
        for f, turns in (battle.fields or {}).items():
            if f == Field.TRICK_ROOM:
                buf[_FOFF_PSEUDO] = 1.0
                buf[_FOFF_TR_TURNS + max(0, min(turns, 3))] = 1.0
            elif f == Field.GRAVITY:
                buf[_FOFF_PSEUDO + 1] = 1.0
                buf[_FOFF_GRAVITY_T + max(0, min(turns, 3))] = 1.0

        # Hazards (own side)
        own_sc = battle.side_conditions or {}
        buf[_FOFF_HAZARDS_OWN] = 1.0 if SideCondition.STEALTH_ROCK in own_sc else 0.0
        sp = own_sc.get(SideCondition.SPIKES, 0)
        buf[_FOFF_HAZARDS_OWN + 1] = 1.0 if sp >= 1 else 0.0
        buf[_FOFF_HAZARDS_OWN + 2] = 1.0 if sp >= 2 else 0.0
        buf[_FOFF_HAZARDS_OWN + 3] = 1.0 if sp >= 3 else 0.0
        ts = own_sc.get(SideCondition.TOXIC_SPIKES, 0)
        buf[_FOFF_HAZARDS_OWN + 4] = 1.0 if ts >= 1 else 0.0
        buf[_FOFF_HAZARDS_OWN + 5] = 1.0 if ts >= 2 else 0.0
        buf[_FOFF_HAZARDS_OWN + 6] = 1.0 if SideCondition.STICKY_WEB in own_sc else 0.0

        # Hazards (opponent side)
        opp_sc = battle.opponent_side_conditions or {}
        buf[_FOFF_HAZARDS_OPP] = 1.0 if SideCondition.STEALTH_ROCK in opp_sc else 0.0
        sp = opp_sc.get(SideCondition.SPIKES, 0)
        buf[_FOFF_HAZARDS_OPP + 1] = 1.0 if sp >= 1 else 0.0
        buf[_FOFF_HAZARDS_OPP + 2] = 1.0 if sp >= 2 else 0.0
        buf[_FOFF_HAZARDS_OPP + 3] = 1.0 if sp >= 3 else 0.0
        ts = opp_sc.get(SideCondition.TOXIC_SPIKES, 0)
        buf[_FOFF_HAZARDS_OPP + 4] = 1.0 if ts >= 1 else 0.0
        buf[_FOFF_HAZARDS_OPP + 5] = 1.0 if ts >= 2 else 0.0
        buf[_FOFF_HAZARDS_OPP + 6] = 1.0 if SideCondition.STICKY_WEB in opp_sc else 0.0

        # Screens (own)
        ls = own_sc.get(SideCondition.LIGHT_SCREEN, 0)
        ref = own_sc.get(SideCondition.REFLECT, 0)
        buf[_FOFF_SCREENS_OWN] = 1.0 if ls > 0 else 0.0
        buf[_FOFF_SCREENS_OWN + 1] = min(ls / 5.0, 1.0)
        buf[_FOFF_SCREENS_OWN + 2] = 1.0 if ref > 0 else 0.0
        buf[_FOFF_SCREENS_OWN + 3] = min(ref / 5.0, 1.0)

        # Screens (opponent)
        ls = opp_sc.get(SideCondition.LIGHT_SCREEN, 0)
        ref = opp_sc.get(SideCondition.REFLECT, 0)
        buf[_FOFF_SCREENS_OPP] = 1.0 if ls > 0 else 0.0
        buf[_FOFF_SCREENS_OPP + 1] = min(ls / 5.0, 1.0)
        buf[_FOFF_SCREENS_OPP + 2] = 1.0 if ref > 0 else 0.0
        buf[_FOFF_SCREENS_OPP + 3] = min(ref / 5.0, 1.0)

        # Turn number bin
        turn = battle.turn or 1
        turn_idx = int(np.sum(turn >= _TURN_THRESHOLDS))
        buf[_FOFF_TURN_BIN + max(0, min(turn_idx, 9))] = 1.0

        # Fainted counts
        own_fainted = sum(1 for p in battle.team.values() if p.fainted)
        opp_fainted = sum(1 for p in battle.opponent_team.values() if p.fainted)
        buf[_FOFF_FAINTED] = own_fainted / 6.0
        buf[_FOFF_FAINTED + 1] = opp_fainted / 6.0

        # Toxic counts (rough estimate)
        active = battle.active_pokemon
        if active and active.status == Status.TOX:
            tox_count = min(getattr(active, 'status_counter', 1) or 1, 5)
            idx = min(tox_count, 4)
            buf[_FOFF_TOXIC_OWN + idx] = 1.0
        else:
            buf[_FOFF_TOXIC_OWN] = 1.0

        opp_active = battle.opponent_active_pokemon
        if opp_active and opp_active.status == Status.TOX:
            buf[_FOFF_TOXIC_OPP + 1] = 1.0  # rough estimate
        else:
            buf[_FOFF_TOXIC_OPP] = 1.0

        # Tailwind
        buf[_FOFF_TAILWIND] = 1.0 if SideCondition.TAILWIND in own_sc else 0.0
        buf[_FOFF_TAILWIND + 1] = 1.0 if SideCondition.TAILWIND in opp_sc else 0.0

        # Safeguard
        buf[_FOFF_SAFEGUARD] = 1.0 if SideCondition.SAFEGUARD in own_sc else 0.0
        buf[_FOFF_SAFEGUARD + 1] = 1.0 if SideCondition.SAFEGUARD in opp_sc else 0.0

        # Mist
        buf[_FOFF_MIST] = 1.0 if SideCondition.MIST in own_sc else 0.0
        buf[_FOFF_MIST + 1] = 1.0 if SideCondition.MIST in opp_sc else 0.0

        return buf

    def build_obs(self, battle: AbstractBattle) -> dict:
        """
        Build pokejax observation from poke-env Battle.

        Returns dict with:
          int_ids: np.ndarray (15, 8) int32
          float_feats: np.ndarray (15, 394) float32
          legal_mask: np.ndarray (10,) float32
        """
        int_ids_list = []
        float_feats_list = []

        available_moves = battle.available_moves
        available_switches = battle.available_switches

        # Token 0: field
        field_feats = np.zeros(FLOAT_DIM, dtype=np.float32)
        field_feats[:FIELD_DIM] = self._encode_field(battle)
        int_ids_list.append(np.zeros(INT_IDS_PER_TOKEN, dtype=np.int32))
        float_feats_list.append(field_feats)

        # Tokens 1-6: own team
        own_active = battle.active_pokemon
        own_team = list(battle.team.values())
        # Reorder: active first, then reserve (alive), then fainted
        ordered_own = []
        if own_active:
            ordered_own.append(own_active)
        for p in own_team:
            if p != own_active and not p.fainted:
                ordered_own.append(p)
        for p in own_team:
            if p != own_active and p.fainted:
                ordered_own.append(p)

        for slot in range(6):
            if slot < len(ordered_own):
                p = ordered_own[slot]
                is_active = (p == own_active)
                ii, ff = self._encode_pokemon(
                    p, is_own=True, is_active=is_active, slot=slot,
                    available_moves=available_moves if is_active else None,
                )
            else:
                ii = np.zeros(INT_IDS_PER_TOKEN, dtype=np.int32)
                ff = np.zeros(FLOAT_DIM, dtype=np.float32)
            int_ids_list.append(ii)
            float_feats_list.append(ff)

        # Tokens 7-12: opponent team
        opp_active = battle.opponent_active_pokemon
        opp_team = list(battle.opponent_team.values())
        ordered_opp = []
        if opp_active:
            ordered_opp.append(opp_active)
        for p in opp_team:
            if p != opp_active and not p.fainted:
                ordered_opp.append(p)
        for p in opp_team:
            if p != opp_active and p.fainted:
                ordered_opp.append(p)

        for slot in range(6):
            if slot < len(ordered_opp):
                p = ordered_opp[slot]
                is_active = (p == opp_active)
                ii, ff = self._encode_pokemon(
                    p, is_own=False, is_active=is_active, slot=slot,
                )
            else:
                ii = np.zeros(INT_IDS_PER_TOKEN, dtype=np.int32)
                ff = np.zeros(FLOAT_DIM, dtype=np.float32)
            int_ids_list.append(ii)
            float_feats_list.append(ff)

        # Tokens 13-14: actor/critic queries
        for _ in range(2):
            int_ids_list.append(np.zeros(INT_IDS_PER_TOKEN, dtype=np.int32))
            float_feats_list.append(np.zeros(FLOAT_DIM, dtype=np.float32))

        int_ids = np.stack(int_ids_list)      # (15, 8)
        float_feats = np.stack(float_feats_list)  # (15, 394)

        # Legal mask
        legal_mask = np.zeros(N_ACTIONS, dtype=np.float32)
        for i in range(min(len(available_moves), 4)):
            legal_mask[i] = 1.0
        for i in range(min(len(available_switches), 6)):
            legal_mask[4 + i] = 1.0
        if legal_mask.sum() == 0:
            legal_mask[0] = 1.0  # fallback

        return {
            "int_ids": int_ids,
            "float_feats": float_feats,
            "legal_mask": legal_mask,
        }


# ---------------------------------------------------------------------------
# poke-env Player class
# ---------------------------------------------------------------------------

class PokejaxPlayer(Player):
    """
    Pokemon Showdown player using the pokejax PokeTransformer model.

    Usage:
        player = PokejaxPlayer(
            checkpoint_path="checkpoints/bc_final.pkl",
            gen=4,
        )
    """

    def __init__(
        self,
        checkpoint_path: str,
        gen: int = 4,
        temperature: float = 0.0,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.verbose = verbose

        # Load tables and obs bridge
        self.tables = load_tables(gen)
        self.obs_bridge = ObsBridge(self.tables)

        # Load model and checkpoint
        self.model = PokeTransformer()
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        self.params = ckpt["params"]

        # JIT compile the forward pass
        @jax.jit
        def _forward(params, int_ids, float_feats, legal_mask):
            log_probs, value_probs, value = self.model.apply(
                params,
                int_ids[None],  # add batch dim
                float_feats[None],
                legal_mask[None],
            )
            return log_probs[0], value[0]

        self._forward = _forward

        # Warm up JIT
        dummy_int = jnp.zeros((N_TOKENS, INT_IDS_PER_TOKEN), dtype=jnp.int32)
        dummy_float = jnp.zeros((N_TOKENS, FLOAT_DIM), dtype=jnp.float32)
        dummy_mask = jnp.ones(N_ACTIONS, dtype=jnp.float32)
        _ = self._forward(self.params, dummy_int, dummy_float, dummy_mask)

    def choose_move(self, battle: AbstractBattle):
        """Choose a move for the current turn."""
        # Build observation
        obs = self.obs_bridge.build_obs(battle)

        # Convert to JAX arrays
        int_ids = jnp.array(obs["int_ids"])
        float_feats = jnp.array(obs["float_feats"])
        legal_mask = jnp.array(obs["legal_mask"])

        # Forward pass
        log_probs, value = self._forward(self.params, int_ids, float_feats, legal_mask)
        log_probs = np.array(log_probs)
        probs = np.exp(log_probs)

        # Select action
        if self.temperature > 0:
            # Sample from softmax with temperature
            scaled = log_probs / self.temperature
            scaled -= scaled.max()
            p = np.exp(scaled)
            p *= np.array(obs["legal_mask"])
            p /= p.sum() + 1e-8
            action = np.random.choice(N_ACTIONS, p=p)
        else:
            # Greedy
            masked_probs = probs * np.array(obs["legal_mask"])
            action = int(np.argmax(masked_probs))

        if self.verbose:
            legal = np.where(obs["legal_mask"])[0]
            print(f"  Turn {battle.turn}: value={float(value):.3f}")
            for a in sorted(legal, key=lambda x: -probs[x]):
                marker = " <--" if a == action else ""
                print(f"    action {a}: {probs[a]*100:.1f}%{marker}")

        # Map action to poke-env order
        available_moves = battle.available_moves
        available_switches = battle.available_switches

        if action < 4 and action < len(available_moves):
            return self.create_order(available_moves[action])
        elif action >= 4:
            switch_idx = action - 4
            if switch_idx < len(available_switches):
                return self.create_order(available_switches[switch_idx])

        # Fallback: pick first legal action
        if available_moves:
            return self.create_order(available_moves[0])
        if available_switches:
            return self.create_order(available_switches[0])
        return self.choose_default_move()
