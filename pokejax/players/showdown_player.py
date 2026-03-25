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

# poke-env 0.12 moved types from poke_env.environment to poke_env.battle
try:
    from poke_env.battle import (
        AbstractBattle, Pokemon, Move, Weather, Field,
        SideCondition, PokemonType, Status,
    )
except ModuleNotFoundError:
    from poke_env.environment import (  # poke-env 0.8
        AbstractBattle, Pokemon, Move, Weather, Field,
        SideCondition, PokemonType, Status,
    )

try:
    from poke_env.battle import Effect
except (ModuleNotFoundError, ImportError):
    try:
        from poke_env.environment import Effect
    except (ModuleNotFoundError, ImportError):
        Effect = None

from pokejax.rl.model import PokeTransformer, create_model
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
# Must match _VOL_MAP in pokejax/rl/obs_builder.py exactly.
# Indices 11, 13, 14, 19, 22, 24 are unmapped in training (always 0)
# so we must NOT write to them during PS inference either.
_VOLATILE_NAMES = {
    "confusion": 0, "infatuation": 1, "leechseed": 2, "curse": 3,
    "aquaring": 4, "ingrain": 5, "taunt": 6, "encore": 7,
    "flinch": 8, "embargo": 9, "healblock": 10,
    "partiallytrapped": 12,
    "substitute": 15, "yawn": 16, "focusenergy": 17, "charge": 18,
    "torment": 20, "nightmare": 21,
    "mustrecharge": 23, "destinybond": 25, "grudge": 26,
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
        "HEAL_BLOCK": "healblock", "GRUDGE": "grudge",
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

        # Sleep turns — the engine counts DOWN (sleep_turns = remaining turns,
        # 1-3 at onset, reaches 0 on the wake turn). poke-env counts UP
        # (sleep_turns = turns already slept, starts at 0). Invert to estimate
        # remaining turns: remaining ≈ max(0, 3 - turns_slept).
        if status_code == 4:  # SLP
            # poke-env 0.12 renamed sleep_turns → status_counter
            turns_slept = (getattr(pokemon, 'sleep_turns', None)
                           or getattr(pokemon, 'status_counter', 0) or 0)
            remaining = max(0, min(3 - turns_slept, 3))
            buf[_OFF_SLEEP_BIN + remaining] = 1.0
        else:
            buf[_OFF_SLEEP_BIN] = 1.0  # bin 0 = not sleeping

        # Rest turns bin — same inversion. Rest lasts 2 turns in the engine
        # (sleep_turns set to 2 on use, decremented to 1, then 0).
        # poke-env counts up from 0. Estimate remaining = max(0, 2 - turns_slept).
        if status_code == 4:  # SLP
            turns_slept = (getattr(pokemon, 'sleep_turns', None)
                           or getattr(pokemon, 'status_counter', 0) or 0)
            rest_remaining = max(0, min(2 - turns_slept, 2))
            buf[_OFF_REST_BIN + rest_remaining] = 1.0
        else:
            buf[_OFF_REST_BIN] = 1.0  # bin 0 = not resting

        # Substitute HP fraction — match training: actual sub HP / (max_hp/4)
        # poke-env doesn't track sub HP; use 1.0 (full sub) as best estimate
        # since most subs are at full HP when first observed.
        # Training: sub_data_raw / max(1, floor(max_hp/4))
        has_sub = False
        for eff, _ in effects.items():
            vol_name = _EFFECT_TO_VOLATILE.get(eff)
            if vol_name == "substitute":
                has_sub = True
                # Use 1.0 (full substitute) — closer to training avg than 0.25
                buf[_OFF_SUB_FRAC] = 1.0
                break

        # Force trapped (partially trapped by moves like Wrap, Fire Spin, etc.)
        for eff, _ in effects.items():
            vol_name = _EFFECT_TO_VOLATILE.get(eff)
            if vol_name == "partiallytrapped":
                buf[_OFF_FORCE_TRAP] = 1.0
                break

        # Move disabled flags (4 dims)
        if is_own and is_active and available_moves is not None:
            for i, m in enumerate(moves_ordered[:4]):
                if m is not None and hasattr(m, 'is_disabled') and m.is_disabled:
                    buf[_OFF_MOV_DIS + i] = 1.0

        # Confusion bin (4 dims) — match training: one_hot(clip(conf_data, 0, 3))
        conf_found = False
        for eff, turns in effects.items():
            vol_name = _EFFECT_TO_VOLATILE.get(eff)
            if vol_name == "confusion":
                conf_t = max(0, min(turns if isinstance(turns, int) else 0, 3))
                buf[_OFF_CONF_BIN + conf_t] = 1.0
                conf_found = True
                break
        if not conf_found:
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
        # poke-env 0.8: Effect.PERISH_SONG with turns remaining as value
        # poke-env 0.12: PERISH3/PERISH2/PERISH1/PERISH0 separate effects
        perish_count = 0
        if Effect is not None:
            # 0.12 style: PERISH3=3 turns, PERISH2=2, PERISH1=1, PERISH0=about to faint
            for n in (3, 2, 1, 0):
                _pe = getattr(Effect, f'PERISH{n}', None)
                if _pe is not None and _pe in effects:
                    perish_count = n
                    break
            else:
                # 0.8 fallback
                _perish_eff = getattr(Effect, 'PERISH_SONG', None)
                if _perish_eff is not None:
                    for eff, turns in effects.items():
                        if eff == _perish_eff:
                            perish_count = max(0, min(int(turns) if isinstance(turns, int) else 3, 3))
                            break
        buf[_OFF_PERISH_BIN + perish_count] = 1.0

        # Protect count — match training: min(prot_data, 4) / 4
        # Use externally-set counter if available (set by PokejaxPlayer)
        protect_count = getattr(self, '_protect_counter_value', 0)
        if is_own and is_active:
            buf[_OFF_PROTECT] = min(protect_count, 4) / 4.0
        else:
            buf[_OFF_PROTECT] = 0.0

        # Locked move (choice item lock or outrage/petal dance)
        # Match training: checks VOL_LOCKEDMOVE | VOL_CHOICELOCK bits
        # PS-side approximation: locked if only 1 move available but mon knows more
        if is_own and is_active:
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

        # Trick room / gravity / wonder room
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
            elif f == Field.WONDER_ROOM:
                buf[_FOFF_PSEUDO + 2] = 1.0

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

        # Toxic counts — match training: one_hot(bin(status_turns), 5)
        # Training bins: 0=none, 1=count1, 2=count2, 3=count3-4, 4=count5+
        active = battle.active_pokemon
        if active and active.status == Status.TOX:
            tox_count = getattr(active, 'status_counter', 0)
            if tox_count is None:
                tox_count = 0
            tox_count = max(0, min(tox_count, 5))
            if tox_count <= 0:
                idx = 0
            elif tox_count == 1:
                idx = 1
            elif tox_count == 2:
                idx = 2
            elif tox_count <= 4:
                idx = 3
            else:
                idx = 4
            buf[_FOFF_TOXIC_OWN + idx] = 1.0
        else:
            buf[_FOFF_TOXIC_OWN] = 1.0

        opp_active = battle.opponent_active_pokemon
        if opp_active and opp_active.status == Status.TOX:
            # For opponent, poke-env may track status_counter
            opp_tox = getattr(opp_active, 'status_counter', 0)
            if opp_tox is None:
                opp_tox = 0
            opp_tox = max(0, min(opp_tox, 5))
            if opp_tox <= 0:
                idx = 0
            elif opp_tox == 1:
                idx = 1
            elif opp_tox == 2:
                idx = 2
            elif opp_tox <= 4:
                idx = 3
            else:
                idx = 4
            buf[_FOFF_TOXIC_OPP + idx] = 1.0
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

        # poke-env sequencing fix: parse_request() runs before |switch| sets
        # the active pokemon, so available_moves can be empty. Reconstruct
        # from the raw request data to respect choice-lock and disabled moves.
        if not available_moves and battle.active_pokemon and battle.active_pokemon.moves:
            last_req = getattr(battle, '_last_request', None)
            active_req = None
            if last_req and 'active' in last_req:
                active_req = last_req['active'][0]
            if active_req:
                available_moves = battle.active_pokemon.available_moves_from_request(
                    active_req
                )
            else:
                available_moves = list(battle.active_pokemon.moves.values())[:4]
        if not available_switches and battle.active_pokemon:
            available_switches = [
                p for p in battle.team.values()
                if p is not battle.active_pokemon and not p.fainted
            ]

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
        # poke-env's active_pokemon and available_moves can be out of sync:
        #   - After switch-in: active_pokemon is new but available_moves may lag
        #   - All PP depleted: available_moves = [Struggle], not in pokemon.moves
        #   - Timing: request parsed before |switch| updates active_pokemon
        # We try three sources in order:
        #   1. available_moves if they match active_pokemon (normal case)
        #   2. available_moves_from_request — server's authoritative move list
        #   3. active_pokemon.moves — best fallback for observation correctness
        real_active = own_active

        def _moves_match(pokemon, avail):
            """Check if ALL available moves belong to this Pokemon's known moves."""
            if not pokemon or not pokemon.moves:
                return False
            p_names = set(m.id for m in pokemon.moves.values())
            p_names_hp = set()
            for n in p_names:
                p_names_hp.add(n)
                if n.startswith('hiddenpower'):
                    p_names_hp.add('hiddenpower')
            for m in avail:
                n = m.id
                norm = n if not n.startswith('hiddenpower') else 'hiddenpower'
                if n not in p_names and norm not in p_names_hp:
                    return False
            return True

        own_move_list = []
        if available_moves:
            if _moves_match(own_active, available_moves):
                # Normal case: available_moves are for the active pokemon
                own_move_list = list(own_active.moves.values())[:4]
            else:
                # Desync: try to get the authoritative move list from the request
                last_req = getattr(battle, '_last_request', None)
                active_req = (last_req['active'][0]
                              if last_req and 'active' in last_req else None)
                reconstructed = []
                if active_req and own_active:
                    try:
                        reconstructed = list(
                            own_active.available_moves_from_request(active_req)
                        )
                    except Exception:
                        pass

                if reconstructed:
                    own_move_list = reconstructed
                    # Update available_moves so legal_mask and action decoder agree
                    available_moves = reconstructed
                    print(f"  [DESYNC] {own_active.species if own_active else '?'}: "
                          f"available_moves mismatch, reconstructed from request: "
                          f"{[m.id for m in own_move_list]}")
                else:
                    # Request reconstruction failed.
                    # If active pokemon has known moves, use them — they give the
                    # correct observation AND are legal for a freshly switched-in
                    # pokemon (no choice-lock yet). Also update available_moves
                    # so the legal_mask and action decoder agree.
                    if own_active and own_active.moves:
                        own_move_list = list(own_active.moves.values())[:4]
                        available_moves = own_move_list  # sync for legal_mask
                        print(f"  [DESYNC] {own_active.species}: "
                              f"stale available_moves={[m.id for m in available_moves[:4]]}, "
                              f"no request — using pokemon.moves: "
                              f"{[m.id for m in own_move_list]}")
                    else:
                        # Last resort: available_moves (might be stale but
                        # at least gives a legal action)
                        own_move_list = list(available_moves)[:4]
                        print(f"  [DESYNC] {own_active.species if own_active else '?'}: "
                              f"stale available_moves, no request, no pokemon.moves — "
                              f"using available_moves: {[m.id for m in own_move_list]}")
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
        # NOTE: We do NOT mask hazards at max layers or substitute when one exists.
        # The model learns these waste a turn (matching both training and PS behavior).
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

        # Safety: ensure the active pokemon's slot is NEVER legal for switching.
        # poke-env's available_switches can include the newly active pokemon
        # during the switch-in turn (timing issue). Silently remove it.
        active_pokemon = battle.active_pokemon
        for slot in range(6):
            p = own_team[slot]
            if p is not None and p is active_pokemon:
                legal_mask[4 + slot] = 0.0

        if legal_mask.sum() == 0:
            # Should not happen after desync repair above, but log if it does
            active_name = battle.active_pokemon.species if battle.active_pokemon else '?'
            print(f"  [BUG] legal_mask all zeros for {active_name}, "
                  f"own_move_list={[m.id for m in own_move_list if m]}, "
                  f"available_moves={[m.id for m in available_moves]}")
            legal_mask[0] = 1.0  # emergency fallback

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

        # Track consecutive protect count per battle for observation accuracy
        # Key: battle_tag, Value: int (consecutive protect count for active mon)
        self._protect_counter: Dict[str, int] = {}
        self._last_action_was_protect: Dict[str, bool] = {}

        # Load tables and obs bridge
        self.tables = load_tables(gen)
        self.obs_bridge = ObsBridge(self.tables)

        # Load model and checkpoint
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        arch = ckpt.get("arch", "transformer")
        self.model = create_model(arch)
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
        try:
            return self._choose_move_impl(battle)
        except Exception as e:
            # Fallback on any error
            if self.verbose:
                print(f"  Error in choose_move: {e}, using default")
            return self.choose_default_move()

    def _is_forced_switch(self, battle: AbstractBattle) -> bool:
        """Detect forced switch: fainted pokemon OR U-turn/Volt Switch/Baton Pass.

        In the pokejax training engine, forced switches after fainting are
        handled transparently (auto-switch to first alive slot). The model
        was never trained on forced-switch-only states, so its switch
        probabilities are meaningless in this situation. We use a value-based
        heuristic instead.

        We check both battle.force_switch (covers U-turn, Volt Switch, Baton
        Pass, etc.) and active.fainted (covers KO'd pokemon).
        """
        # poke-env's force_switch flag covers all forced-switch scenarios:
        # fainted pokemon, U-turn, Volt Switch, Baton Pass, etc.
        if battle.force_switch:
            return True
        active = battle.active_pokemon
        if active is not None and not active.fainted:
            return False
        # Active is fainted or None — check if we have switches available
        available_switches = battle.available_switches
        if not available_switches and active is not None:
            available_switches = [
                p for p in battle.team.values()
                if p is not active and not p.fainted
            ]
        return len(available_switches) > 0

    def _choose_forced_switch(self, battle: AbstractBattle):
        """Choose the best replacement Pokemon using value-based evaluation.

        For each available switch target, we build an observation as if that
        Pokemon were already active, run the model to get a value estimate,
        and pick the switch target with the highest value. This is more
        principled than the model's untrained forced-switch distribution.
        """
        available_switches = battle.available_switches
        own_team = self.obs_bridge._get_stable_team_order(battle, is_own=True)

        if len(available_switches) == 1:
            if self.verbose:
                print(f"  [FORCED SWITCH] Only one option: {available_switches[0].species}")
            return self.create_order(available_switches[0])

        # For each candidate, build obs pretending it's active and get value
        best_pokemon = None
        best_value = -float('inf')
        candidate_values = []

        # Build base obs once for the field/opponent tokens
        base_obs = self.obs_bridge.build_obs(battle)

        for candidate in available_switches:
            # Find candidate's slot in own_team
            candidate_slot = None
            for slot_idx, p in enumerate(own_team):
                if p is not None and p is candidate:
                    candidate_slot = slot_idx
                    break

            if candidate_slot is None:
                continue

            # Modify obs to make this candidate appear as active:
            # Update the float_feats for the candidate's token (slot+1 since
            # token 0 is field) to mark it as active
            modified_float = base_obs["float_feats"].copy()
            modified_int = base_obs["int_ids"].copy()

            # Clear is_active for all own Pokemon tokens (1-6)
            for s in range(6):
                modified_float[1 + s, _OFF_IS_ACTIVE] = 0.0

            # Set is_active for the candidate
            modified_float[1 + candidate_slot, _OFF_IS_ACTIVE] = 1.0

            # Build a legal mask as if this Pokemon were active (all moves legal)
            # This matches what training sees after a forced switch
            sim_mask = np.zeros(N_ACTIONS, dtype=np.float32)
            # Enable all 4 move slots (the model will see a normal state)
            for i in range(4):
                sim_mask[i] = 1.0
            # Enable switches to other alive non-candidate slots
            for s in range(6):
                p = own_team[s]
                if (p is not None and not p.fainted and
                        p is not candidate and s != candidate_slot):
                    sim_mask[4 + s] = 1.0

            # Ensure at least one action is legal
            if sim_mask.sum() == 0:
                sim_mask[0] = 1.0

            # Forward pass to get value
            int_ids = jnp.array(modified_int)
            float_feats = jnp.array(modified_float)
            legal_mask = jnp.array(sim_mask)

            _, value = self._forward(self.params, int_ids, float_feats, legal_mask)
            val = float(value)
            candidate_values.append((candidate, val))

            if val > best_value:
                best_value = val
                best_pokemon = candidate

        if self.verbose:
            print(f"  [FORCED SWITCH] Value-based replacement selection:")
            for pokemon, val in sorted(candidate_values, key=lambda x: -x[1]):
                marker = " <--" if pokemon is best_pokemon else ""
                print(f"    {pokemon.species}: value={val:.3f}{marker}")

        if best_pokemon is not None:
            return self.create_order(best_pokemon)

        # Fallback: first available switch
        return self.create_order(available_switches[0])

    def _choose_move_impl(self, battle: AbstractBattle):
        """Internal move selection logic."""
        available_moves = battle.available_moves
        available_switches = battle.available_switches
        trapped = getattr(battle, 'trapped', False)

        # Repair available_moves if empty or stale (doesn't match active pokemon).
        # Uses available_moves_from_request as the authoritative source — it
        # respects choice-lock (Outrage), disabled moves, Struggle, and trapped.
        active = battle.active_pokemon

        def _avail_matches_active(avail, poke):
            """True if at least one move in avail belongs to poke's known moves."""
            if not poke or not poke.moves or not avail:
                return False
            p_ids = {m.id for m in poke.moves.values()}
            p_ids_norm = p_ids | {'hiddenpower'} if any(
                n.startswith('hiddenpower') for n in p_ids) else p_ids
            for m in avail:
                n = m.id
                norm = 'hiddenpower' if n.startswith('hiddenpower') else n
                if n in p_ids or norm in p_ids_norm:
                    return True
            return False

        def _try_request_reconstruction(poke):
            last_req = getattr(battle, '_last_request', None)
            if not last_req or 'active' not in last_req:
                return [], None
            active_req = last_req['active'][0]
            try:
                return list(poke.available_moves_from_request(active_req)), active_req
            except Exception:
                return [], active_req

        needs_repair = (
            not available_moves
            or (active and active.moves
                and not _avail_matches_active(available_moves, active))
        )

        if needs_repair and active:
            reconstructed, active_req = _try_request_reconstruction(active) if active else ([], None)
            if reconstructed:
                if active_req and active_req.get('trapped'):
                    trapped = True
                reason = "empty" if not available_moves else "stale/desync"
                if self.verbose or reason == "stale/desync":
                    print(f"  Turn {battle.turn}: [{reason.upper()} AVAIL_MOVES] "
                          f"{active.species}: was={[m.id for m in available_moves]}, "
                          f"reconstructed={[m.id for m in reconstructed]}")
                available_moves = reconstructed
            elif active.moves:
                # Reconstruction failed — use active pokemon's known moves.
                # These are the correct moves for any freshly switched-in pokemon
                # (no choice-lock yet). Also covers the stale-available_moves case.
                old = [m.id for m in available_moves]
                available_moves = list(active.moves.values())[:4]
                if self.verbose or (old and set(old) != {m.id for m in available_moves}):
                    label = "stale/desync" if old else "empty"
                    print(f"  Turn {battle.turn}: [{label.upper()} AVAIL_MOVES no request] "
                          f"{active.species}: was={old}, "
                          f"using pokemon.moves={[m.id for m in available_moves]}")
        if not available_switches and battle.active_pokemon:
            available_switches = [
                p for p in battle.team.values()
                if p is not battle.active_pokemon and not p.fainted
            ]
            if self.verbose and available_switches:
                print(f"  Turn {battle.turn}: reconstructed available_switches: "
                      f"{[p.species for p in available_switches]}")

        # If still no moves AND no switches, nothing to decide.
        if not available_moves and not available_switches:
            if self.verbose:
                print(f"  Turn {battle.turn}: no moves or switches available, "
                      f"using default move")
            return self.choose_default_move()

        # Detect forced switch (after faint): use value-based heuristic
        # instead of the model's untrained forced-switch distribution
        if self._is_forced_switch(battle):
            return self._choose_forced_switch(battle)

        # Update protect counter for observation
        btag = battle.battle_tag if hasattr(battle, 'battle_tag') else ""
        if btag not in self._protect_counter:
            self._protect_counter[btag] = 0
            self._last_action_was_protect[btag] = False

        # If last action was NOT protect, reset counter
        if not self._last_action_was_protect.get(btag, False):
            self._protect_counter[btag] = 0

        # Set counter on obs bridge so _encode_pokemon can use it
        self.obs_bridge._protect_counter_value = self._protect_counter[btag]

        # Build observation (also stores _last_own_team and _last_own_move_list)
        obs = self.obs_bridge.build_obs(battle)

        # Re-sync available_moves after build_obs: the PS |request| may have
        # arrived asynchronously while build_obs was running, updating
        # battle.available_moves after our pre-obs repair.  Only use the fresh
        # battle state if it actually belongs to the current active pokemon;
        # otherwise fall back to _last_own_move_list (what the obs/legal_mask
        # were built from) so the action decoder is consistent.
        fresh_avail = list(battle.available_moves)
        if fresh_avail and _avail_matches_active(fresh_avail, battle.active_pokemon):
            available_moves = fresh_avail
        else:
            available_moves = [
                m for m in self.obs_bridge._last_own_move_list if m is not None
            ] or available_moves

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
            own_team = self.obs_bridge._last_own_team
            own_move_list = self.obs_bridge._last_own_move_list
            active_name = battle.active_pokemon.species if battle.active_pokemon else "?"
            opp_name = battle.opponent_active_pokemon.species if battle.opponent_active_pokemon else "?"
            print(f"  Turn {battle.turn}: {active_name} vs {opp_name} | "
                  f"value={float(value):.3f}{trap_str}")
            for a in sorted(legal, key=lambda x: -probs[x]):
                marker = " <--" if a == action else ""
                if a < 4:
                    mname = own_move_list[a].id if a < len(own_move_list) and own_move_list[a] else "?"
                    print(f"    move {a} ({mname}): {probs[a]*100:.1f}%{marker}")
                else:
                    slot = a - 4
                    pname = own_team[slot].species if slot < len(own_team) and own_team[slot] else "?"
                    print(f"    switch {slot} ({pname}): {probs[a]*100:.1f}%{marker}")

        # Map action to poke-env order using stable slot mappings.
        # Try the top action first; if invalid (desync), fall back to next-best.
        own_team = self.obs_bridge._last_own_team
        own_move_list = self.obs_bridge._last_own_move_list
        active_pokemon = battle.active_pokemon
        available_switch_set = set(id(p) for p in available_switches)
        available_move_ids = set(m.id for m in available_moves)

        # Build sorted action list by probability (greedy fallback)
        masked_probs_all = probs * np.array(obs["legal_mask"])
        sorted_actions = list(np.argsort(-masked_probs_all))

        def _track_and_return(order, move_id=None):
            """Track protect usage for counter, then return the order."""
            is_protect = move_id in ("protect", "detect")
            self._last_action_was_protect[btag] = is_protect
            if is_protect:
                self._protect_counter[btag] = self._protect_counter.get(btag, 0) + 1
            return order

        for act in sorted_actions:
            if masked_probs_all[act] <= 0:
                break  # no more legal actions

            if act < 4:
                # Move action
                if act < len(own_move_list) and own_move_list[act] is not None:
                    chosen_move_id = own_move_list[act].id
                    if chosen_move_id in available_move_ids:
                        for m in available_moves:
                            if m.id == chosen_move_id:
                                return _track_and_return(
                                    self.create_order(m), chosen_move_id)
                    elif self.verbose:
                        print(f"  [DESYNC] move {act} ({chosen_move_id}) not in "
                              f"available_moves, trying next action")
            else:
                # Switch action
                slot = act - 4
                if slot < len(own_team) and own_team[slot] is not None:
                    target = own_team[slot]
                    if target is active_pokemon:
                        if self.verbose:
                            print(f"  [DESYNC] switch to active "
                                  f"{active_pokemon.species}, trying next action")
                        continue
                    if id(target) in available_switch_set:
                        return _track_and_return(
                            self.create_order(target), "switch")
                    elif self.verbose:
                        print(f"  [DESYNC] switch to {target.species} not in "
                              f"available_switches, trying next action")

        # Ultimate fallback: pick first available action
        self._last_action_was_protect[btag] = False
        if available_moves:
            return self.create_order(available_moves[0])
        if available_switches:
            return self.create_order(available_switches[0])
        return self.choose_default_move()
