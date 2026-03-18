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
            try:
                priority = getattr(move, 'priority', 0) or 0
            except (KeyError, AttributeError):
                priority = 0

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
            # For opponents, we only know the HP fraction from Showdown
            # Use it directly instead of estimating absolute HP
            hp_frac_raw = pokemon.current_hp_fraction if pokemon.current_hp_fraction is not None else 1.0
            maxhp = 1  # placeholder; hp_frac computed directly below
            hp = 1  # placeholder

        if is_own:
            hp_frac = hp / max(maxhp, 1)
        else:
            hp_frac = hp_frac_raw
        buf[_OFF_HP_FRAC] = hp_frac
        hp_pct = max(0, min(int(hp_frac * 100), 100))
        buf[_OFF_HP_BIN:_OFF_HP_BIN + 10] = 0
        buf[_OFF_HP_BIN + _bin_idx(hp_pct, _HP_THRESHOLDS)] = 1.0

        # Base stats (normalized /255)
        # Model was trained with engine's team pool which now has real stats
        # and the model handles them fine (100% vs random in engine)
        bst = getattr(pokemon, 'base_stats', {}) or {}
        stat_order = ['hp', 'atk', 'def', 'spa', 'spd', 'spe']
        for i, sname in enumerate(stat_order):
            buf[_OFF_BASE_STATS + i] = bst.get(sname, 80) / 255.0

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

        # Types (one-hot, PS uses 0-based: Normal=0, Fire=1, ...)
        types = pokemon.types if hasattr(pokemon, 'types') and pokemon.types else (PokemonType.NORMAL,)
        if types[0] is not None:
            pokejax_type = _TYPE_MAP.get(types[0], 1)  # pokejax 1-based
            ps_type1 = max(0, min(pokejax_type - 1, _N_PS_TYPES - 1))
            buf[_OFF_TYPE1 + ps_type1] = 1.0
        else:
            buf[_OFF_TYPE1 + 0] = 1.0  # Normal fallback
        if len(types) > 1 and types[1] is not None:
            pokejax_type2 = _TYPE_MAP.get(types[1], 0)
            if pokejax_type2 > 0:
                ps_type2 = max(0, min(pokejax_type2 - 1, _N_PS_TYPES - 1))
                buf[_OFF_TYPE2 + ps_type2] = 1.0

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

        # Substitute HP fraction
        has_sub = False
        for eff, _ in effects.items():
            vol_name = _EFFECT_TO_VOLATILE.get(eff)
            if vol_name == "substitute":
                has_sub = True
                # poke-env doesn't track sub HP; assume 25% of max HP remaining
                buf[_OFF_SUB_FRAC] = 0.25
                break

        # Force trapped (partially trapped by moves like Wrap, Fire Spin, etc.)
        for eff, _ in effects.items():
            vol_name = _EFFECT_TO_VOLATILE.get(eff)
            if vol_name == "partiallytrapped":
                buf[_OFF_FORCE_TRAP] = 1.0
                break

        # Move disabled flags (4 dims)
        if is_own and is_active and available_moves is not None:
            # Check which move slots are disabled
            all_moves = list(pokemon.moves.values()) if pokemon.moves else []
            for i, m in enumerate(moves_ordered[:4]):
                if m is not None and hasattr(m, 'is_disabled') and m.is_disabled:
                    buf[_OFF_MOV_DIS + i] = 1.0

        # Confusion bin (4 dims)
        for eff, turns in effects.items():
            vol_name = _EFFECT_TO_VOLATILE.get(eff)
            if vol_name == "confusion":
                conf_t = max(0, min(turns if isinstance(turns, int) else 1, 3))
                buf[_OFF_CONF_BIN + conf_t] = 1.0
                break
        else:
            buf[_OFF_CONF_BIN] = 1.0  # no confusion = bin 0

        # Taunt, encore, yawn flags
        for eff, _ in effects.items():
            vol_name = _EFFECT_TO_VOLATILE.get(eff)
            if vol_name == "taunt":
                buf[_OFF_TAUNT] = 1.0
            elif vol_name == "encore":
                buf[_OFF_ENCORE] = 1.0
            elif vol_name == "yawn":
                buf[_OFF_YAWN] = 1.0

        # Level
        level = pokemon.level if pokemon.level else 100
        buf[_OFF_LEVEL] = level / 100.0

        # Perish count bin (4 dims: 0=none, 1, 2, 3)
        perish_count = 0
        for eff, turns in effects.items():
            vol_name = _EFFECT_TO_VOLATILE.get(eff)
            if vol_name == "perishsong":
                perish_count = max(0, min(turns if isinstance(turns, int) else 3, 3))
                break
        buf[_OFF_PERISH_BIN + perish_count] = 1.0

        # Protect count (normalized)
        # poke-env doesn't track protect count; default 0
        buf[_OFF_PROTECT] = 0.0

        # Locked move (choice item lock or outrage/petal dance)
        if is_own and is_active:
            # If only 1 move is available and we have more moves, likely locked
            if available_moves is not None and len(available_moves) == 1 and len(pokemon.moves) > 1:
                buf[_OFF_LOCKED_MOV] = 1.0

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
        # Defaults: engine always sets bin 0 for these when inactive
        buf[_FOFF_TR_TURNS] = 1.0
        buf[_FOFF_GRAVITY_T] = 1.0
        for f, turns in (battle.fields or {}).items():
            if f == Field.TRICK_ROOM:
                buf[_FOFF_PSEUDO] = 1.0
                buf[_FOFF_TR_TURNS] = 0.0  # clear default
                buf[_FOFF_TR_TURNS + max(0, min(turns, 3))] = 1.0
            elif f == Field.GRAVITY:
                buf[_FOFF_PSEUDO + 1] = 1.0
                buf[_FOFF_GRAVITY_T] = 0.0  # clear default
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

    def _get_stable_team_order(self, battle: AbstractBattle, is_own: bool):
        """Return a stable list of up to 6 Pokemon in fixed slot order.

        poke-env's battle.team dict is keyed by species identifier and
        the order of insertion is preserved (first seen = first entry).
        We use this insertion order as the stable slot assignment so that
        each Pokemon always occupies the same token position throughout
        the battle, matching how pokejax's engine assigns fixed slot
        indices 0-5.
        """
        if is_own:
            team = list(battle.team.values())
        else:
            team = list(battle.opponent_team.values())
        # Pad to 6 slots (unseen Pokemon → None)
        while len(team) < 6:
            team.append(None)
        return team[:6]

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

        # Build stable slot assignments (fixed order, no reordering)
        own_team = self._get_stable_team_order(battle, is_own=True)
        opp_team = self._get_stable_team_order(battle, is_own=False)

        own_active = battle.active_pokemon
        opp_active = battle.opponent_active_pokemon

        # Resolve move list for the active Pokemon.
        # Edge cases (Hidden Power variants, etc.) can cause mismatches between
        # the active Pokemon's known moves and available_moves. We resolve the
        # move list independently — real_active is ALWAYS battle.active_pokemon
        # to stay consistent with the legal mask.
        real_active = own_active
        own_move_list = []
        if available_moves:
            avail_names = set(m.id for m in available_moves)

            def _moves_match(pokemon):
                """Check if ALL available moves are in this Pokemon's known moves.
                Handles Hidden Power variants (e.g., 'hiddenpowerice' vs 'hiddenpower')."""
                if not pokemon or not pokemon.moves:
                    return False
                p_names = set(m.id for m in pokemon.moves.values())
                # Also add normalized variants for Hidden Power matching
                p_names_hp = set()
                for n in p_names:
                    p_names_hp.add(n)
                    if n.startswith('hiddenpower'):
                        p_names_hp.add('hiddenpower')
                for n in avail_names:
                    norm = n if not n.startswith('hiddenpower') else 'hiddenpower'
                    if n not in p_names and norm not in p_names_hp:
                        return False
                return True

            # First check if battle.active_pokemon's moves match
            if _moves_match(own_active):
                own_move_list = list(own_active.moves.values())[:4]
            else:
                # Move mismatch — find the pokemon whose moves match available_moves
                # for move list ordering, but do NOT change real_active (must stay
                # consistent with legal mask to avoid switching to active pokemon).
                move_source = None
                best_match = None
                best_overlap = 0
                for p in own_team:
                    if p is not None and p.moves and not p.fainted:
                        if _moves_match(p):
                            move_source = p
                            own_move_list = list(p.moves.values())[:4]
                            break
                        p_names = set(m.id for m in p.moves.values())
                        overlap = len(avail_names & p_names)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_match = p

                if not own_move_list:
                    if best_match is not None:
                        move_source = best_match
                        own_move_list = list(best_match.moves.values())[:4]
                    elif own_active and own_active.moves:
                        own_move_list = list(own_active.moves.values())[:4]
                    else:
                        # Last resort: use available_moves directly as the move list
                        own_move_list = list(available_moves)[:4]

                # If we found a different move source, log it but keep real_active
                # as battle.active_pokemon for observation/mask consistency
                if move_source is not None and move_source is not own_active:
                    print(f"  [WARN] Move mismatch: poke-env active="
                          f"{own_active.species if own_active else None}, "
                          f"moves match={move_source.species}")
        elif own_active and own_active.moves:
            own_move_list = list(own_active.moves.values())[:4]

        # Tokens 1-6: own team (fixed slot order, no reordering)
        for slot in range(6):
            p = own_team[slot]
            if p is not None:
                is_active = (p is real_active)
                # For active Pokemon, pass moves in stable slot order
                slot_moves = own_move_list if is_active else None
                ii, ff = self._encode_pokemon(
                    p, is_own=True, is_active=is_active, slot=slot,
                    available_moves=slot_moves,
                )
            else:
                ii = np.zeros(INT_IDS_PER_TOKEN, dtype=np.int32)
                ff = np.zeros(FLOAT_DIM, dtype=np.float32)
            int_ids_list.append(ii)
            float_feats_list.append(ff)

        # Tokens 7-12: opponent team (fixed slot order)
        for slot in range(6):
            p = opp_team[slot]
            if p is not None:
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

        # Legal mask — built from slot positions, matching pokejax engine
        legal_mask = np.zeros(N_ACTIONS, dtype=np.float32)

        # Move actions 0-3: legal if the move (by slot order) is in available_moves
        # Use move.id string matching (not object identity) to handle poke-env
        # creating new Move objects for variants like Hidden Power Ice
        available_move_names = set(m.id for m in available_moves)
        for i, m in enumerate(own_move_list[:4]):
            if m is not None and m.id in available_move_names:
                legal_mask[i] = 1.0

        # Switch actions 4-9: legal if Pokemon at that slot is in available_switches
        # Pokemon objects are persistent, so identity matching works here
        available_switch_ids = set(id(p) for p in available_switches)
        for slot in range(6):
            p = own_team[slot]
            if p is not None and id(p) in available_switch_ids:
                legal_mask[4 + slot] = 1.0

        # Safety: ensure the active pokemon's slot is NEVER legal for switching,
        # even if poke-env's available_switches is stale or inconsistent.
        active_pokemon = battle.active_pokemon
        for slot in range(6):
            p = own_team[slot]
            if p is not None and p is active_pokemon and legal_mask[4 + slot] > 0:
                print(f"  [BUG] legal_mask included active pokemon "
                      f"{active_pokemon.species} at slot {slot} — removing")
                legal_mask[4 + slot] = 0.0

        if legal_mask.sum() == 0:
            legal_mask[0] = 1.0  # fallback

        # Store mappings for action decoding in choose_move
        self._last_own_team = own_team
        self._last_own_move_list = own_move_list

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

    Just overrides choose_move — lets poke-env handle all protocol/messaging.

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

    def _stale_request_type(self, battle: AbstractBattle) -> int:
        """Detect when poke-env calls choose_move with stale request data.

        After a switch, poke-env processes |switch| (updating active_pokemon)
        before the new |request| arrives. This means available_moves still
        belongs to the PREVIOUS active pokemon.

        Returns:
            0: not stale
            1: move mismatch (available_moves from different pokemon)
            2: forced switch stale (active pokemon in available_switches)
        """
        active = battle.active_pokemon
        if not active:
            return 0
        available_moves = battle.available_moves
        available_switches = battle.available_switches
        # Case 1: any available move doesn't belong to active pokemon
        if available_moves and active.moves:
            active_move_ids = set(m.id for m in active.moves.values())
            # Add Hidden Power base form for variant matching
            active_hp = set(active_move_ids)
            for mid in active_move_ids:
                if mid.startswith('hiddenpower'):
                    active_hp.add('hiddenpower')
            for m in available_moves:
                mid = m.id
                norm = mid if not mid.startswith('hiddenpower') else 'hiddenpower'
                if mid not in active_hp and norm not in active_hp:
                    return 1
        # Case 2: forced switch but active pokemon is in available_switches
        if not available_moves and available_switches:
            if active in available_switches:
                return 2
        return 0

    async def _handle_battle_request(
        self,
        battle: AbstractBattle,
        from_teampreview_request: bool = False,
        maybe_default_order: bool = False,
    ):
        """Override to handle stale requests after switches.

        When poke-env processes |switch| + |turn| before the new |request|
        arrives, the battle state is inconsistent (active_pokemon updated
        from |switch| msg but available_moves from old request). We send a
        deliberately wrong move so PS rejects it and sends the correct
        request, avoiding running the model on garbage data.
        """
        if (not from_teampreview_request and not maybe_default_order
                and not battle.teampreview):
            stale_type = self._stale_request_type(battle)
            if stale_type == 1:
                # Move mismatch: send a stale move that PS will reject
                if self.verbose:
                    print(f"  [STALE] turn={battle.turn}: active="
                          f"{battle.active_pokemon.species} — sending "
                          f"rejectable move")
                available_moves = battle.available_moves
                if available_moves:
                    message = self.create_order(available_moves[0]).message
                    await self.ps_client.send_message(
                        message, battle.battle_tag)
                    return
            elif stale_type == 2:
                # Forced switch stale: switch to active (PS rejects)
                if self.verbose:
                    print(f"  [STALE] turn={battle.turn}: active="
                          f"{battle.active_pokemon.species} — sending "
                          f"rejectable switch")
                message = self.create_order(battle.active_pokemon).message
                await self.ps_client.send_message(message, battle.battle_tag)
                return
        await super()._handle_battle_request(
            battle,
            from_teampreview_request=from_teampreview_request,
            maybe_default_order=maybe_default_order,
        )

    def choose_move(self, battle: AbstractBattle):
        """Choose a move for the current turn."""
        try:
            return self._choose_move_impl(battle)
        except Exception as e:
            # Fallback on any error
            if self.verbose:
                print(f"  Error in choose_move: {e}, using default")
            return self.choose_default_move()

    def _choose_move_impl(self, battle: AbstractBattle):
        """Internal move selection logic."""
        available_moves = battle.available_moves
        available_switches = battle.available_switches

        # Check if trapped (can't switch)
        trapped = battle.trapped if hasattr(battle, 'trapped') else False

        # Build observation (also stores _last_own_team and _last_own_move_list)
        obs = self.obs_bridge.build_obs(battle)

        # Override legal mask: if trapped, disable all switch actions
        if trapped:
            obs["legal_mask"][4:] = 0.0
            # Ensure at least one move is legal
            if obs["legal_mask"][:4].sum() == 0 and available_moves:
                available_move_names = set(m.id for m in available_moves)
                for i, m in enumerate(self.obs_bridge._last_own_move_list[:4]):
                    if m is not None and m.id in available_move_names:
                        obs["legal_mask"][i] = 1.0
            if obs["legal_mask"].sum() == 0:
                obs["legal_mask"][0] = 1.0

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
            scaled = log_probs / self.temperature
            scaled -= scaled.max()
            p = np.exp(scaled)
            p *= np.array(obs["legal_mask"])
            p /= p.sum() + 1e-8
            action = np.random.choice(N_ACTIONS, p=p)
        else:
            masked_probs = probs * np.array(obs["legal_mask"])
            action = int(np.argmax(masked_probs))

        if self.verbose:
            legal = np.where(obs["legal_mask"])[0]
            trap_str = " [TRAPPED]" if trapped else ""
            print(f"  Turn {battle.turn}: value={float(value):.3f}{trap_str}")
            for a in sorted(legal, key=lambda x: -probs[x]):
                marker = " <--" if a == action else ""
                print(f"    action {a}: {probs[a]*100:.1f}%{marker}")

        # Map action to poke-env order using stable slot mappings
        own_team = self.obs_bridge._last_own_team
        own_move_list = self.obs_bridge._last_own_move_list
        active_pokemon = battle.active_pokemon

        if action < 4:
            # Move action: action i = move slot i from pokemon.moves dict order
            if action < len(own_move_list) and own_move_list[action] is not None:
                chosen_move_id = own_move_list[action].id
                # Find the matching Move in available_moves by string ID
                # (poke-env may create new objects for variants like Hidden Power)
                for m in available_moves:
                    if m.id == chosen_move_id:
                        return self.create_order(m)
            # Fallback: try to find any legal move
            if available_moves:
                return self.create_order(available_moves[0])
        else:
            # Switch action: action 4+slot = switch to Pokemon at team slot
            slot = action - 4
            if slot < len(own_team) and own_team[slot] is not None:
                target_pokemon = own_team[slot]
                # Safety: never switch to the active pokemon
                if target_pokemon is active_pokemon:
                    if self.verbose:
                        print(f"  [BUG] Model chose switch to active "
                              f"{active_pokemon.species} (slot {slot}), "
                              f"falling back")
                elif target_pokemon in available_switches:
                    return self.create_order(target_pokemon)
            # Fallback: try to find any legal switch
            if available_switches:
                return self.create_order(available_switches[0])

        # Ultimate fallback
        if available_moves:
            return self.create_order(available_moves[0])
        if available_switches:
            return self.create_order(available_switches[0])
        return self.choose_default_move()
