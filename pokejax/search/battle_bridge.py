"""
Convert a poke-env Battle object into a pokejax BattleState + RevealState
so the GPU engine can simulate future turns for search.

Delegates all name→ID lookups to the existing ObsBridge from showdown_player,
avoiding any duplication.  Only the BattleState assembly logic is new here
(ObsBridge builds observations; this module builds engine state).

Imperfect information handling:
  - Own team: fully known, mapped exactly.
  - Opponent team: only revealed info is used.
    * Unrevealed Pokemon slots → species_id=0, placeholder stats, fainted=True.
    * Unrevealed moves → move_id=-1 (empty).
    * HP estimated from hp_fraction × level-based max-HP formula.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import jax.numpy as jnp

try:
    from poke_env.battle import (
        AbstractBattle, Pokemon, Weather, Field, SideCondition, Status,
    )
except ModuleNotFoundError:
    from poke_env.environment import (
        AbstractBattle, Pokemon, Weather, Field, SideCondition, Status,
    )

try:
    from poke_env.battle import Effect
except (ModuleNotFoundError, ImportError):
    try:
        from poke_env.environment import Effect
    except (ModuleNotFoundError, ImportError):
        Effect = None

from pokejax.types import (
    BattleState, FieldState, RevealState,
    MAX_SIDE_CONDS, MAX_VOLATILES,
    SC_SPIKES, SC_TOXICSPIKES, SC_STEALTHROCK, SC_STICKYWEB,
    SC_REFLECT, SC_LIGHTSCREEN, SC_AURORAVEIL, SC_TAILWIND,
    SC_SAFEGUARD, SC_MIST,
    VOL_CONFUSED, VOL_FLINCH, VOL_PARTIALLY_TRAPPED, VOL_SEEDED,
    VOL_SUBSTITUTE, VOL_ENCORE, VOL_TAUNT, VOL_TORMENT,
    VOL_INGRAIN, VOL_AQUARINGTARGET, VOL_FOCUSENERGY,
    VOL_CURSE, VOL_NIGHTMARE, VOL_ATTRACT, VOL_YAWN,
    VOL_DESTINYBOND, VOL_GRUDGE, VOL_EMBARGO, VOL_HEALBLOCK,
)

# Import the same enum maps that showdown_player uses
from pokejax.players.showdown_player import _STATUS_MAP, _TYPE_MAP, _WEATHER_MAP
from pokejax.data.extractor import NATURE_NAMES

# Nature name → index lookup (normalized lowercase, matches poke-env's pokemon.nature)
import re as _re
_NATURE_LOOKUP = {_re.sub(r'[^a-z0-9]', '', n.lower()): i for i, n in enumerate(NATURE_NAMES)}

# ── Effect enum → volatile bit index ─────────────────────────────────
_EFFECT_TO_VOL_BIT = {}
if Effect is not None:
    _VOL_BIT_MAP = {
        "CONFUSION": VOL_CONFUSED, "ATTRACT": VOL_ATTRACT,
        "LEECH_SEED": VOL_SEEDED, "CURSE": VOL_CURSE,
        "AQUA_RING": VOL_AQUARINGTARGET, "INGRAIN": VOL_INGRAIN,
        "TAUNT": VOL_TAUNT, "ENCORE": VOL_ENCORE,
        "FLINCH": VOL_FLINCH, "EMBARGO": VOL_EMBARGO,
        "FOCUS_ENERGY": VOL_FOCUSENERGY, "SUBSTITUTE": VOL_SUBSTITUTE,
        "YAWN": VOL_YAWN, "TORMENT": VOL_TORMENT,
        "NIGHTMARE": VOL_NIGHTMARE, "DESTINY_BOND": VOL_DESTINYBOND,
        "HEAL_BLOCK": VOL_HEALBLOCK,
    }
    for _a, _b in _VOL_BIT_MAP.items():
        try:
            _EFFECT_TO_VOL_BIT[getattr(Effect, _a)] = _b
        except AttributeError:
            pass

# ── Side condition mapping ────────────────────────────────────────────
_SC_MAP = {
    SideCondition.STEALTH_ROCK: SC_STEALTHROCK,
    SideCondition.SPIKES: SC_SPIKES,
    SideCondition.TOXIC_SPIKES: SC_TOXICSPIKES,
    SideCondition.LIGHT_SCREEN: SC_LIGHTSCREEN,
    SideCondition.REFLECT: SC_REFLECT,
    SideCondition.TAILWIND: SC_TAILWIND,
    SideCondition.SAFEGUARD: SC_SAFEGUARD,
    SideCondition.MIST: SC_MIST,
}
try:
    _SC_MAP[SideCondition.STICKY_WEB] = SC_STICKYWEB
except AttributeError:
    pass
try:
    _SC_MAP[SideCondition.AURORA_VEIL] = SC_AURORAVEIL
except AttributeError:
    pass


class BattleBridge:
    """Converts a poke-env Battle → pokejax (BattleState, RevealState).

    Takes an ObsBridge instance to reuse its name→ID lookups.
    """

    def __init__(self, obs_bridge):
        """
        Parameters
        ----------
        obs_bridge : ObsBridge
            An initialized ObsBridge (from showdown_player.py).
            All species/move/ability/item ID lookups are delegated to it.
        """
        self.obs = obs_bridge

    # ── Per-Pokemon ───────────────────────────────────────────────────

    def _pokemon_to_engine(self, pokemon: Optional[Pokemon], is_own: bool) -> dict:
        """Convert one poke-env Pokemon → dict of numpy arrays for BattleState."""
        if pokemon is None:
            return {
                'species_id': np.int16(0), 'ability_id': np.int16(0),
                'item_id': np.int16(0),
                'types': np.array([1, 0], dtype=np.int8),
                'base_stats': np.full(6, 80, dtype=np.int16),
                'hp': np.int16(0), 'max_hp': np.int16(1),
                'boosts': np.zeros(7, dtype=np.int8),
                'move_ids': np.full(4, -1, dtype=np.int16),
                'move_pp': np.zeros(4, dtype=np.int8),
                'move_max_pp': np.zeros(4, dtype=np.int8),
                'move_disabled': np.zeros(4, dtype=np.bool_),
                'status': np.int8(0), 'status_turns': np.int8(0),
                'sleep_turns': np.int8(0),
                'volatiles': np.uint32(0),
                'volatile_data': np.zeros(MAX_VOLATILES, dtype=np.int8),
                'is_active': False, 'fainted': True,
                'last_move_id': np.int16(-1),
                'level': np.int8(100), 'gender': np.int8(0),
                'nature_id': np.int8(0), 'weight_hg': np.int16(500),
            }

        # Delegate name→ID lookups to ObsBridge
        species_id = self.obs._species_id(pokemon)
        ability_id = self.obs._ability_id(pokemon) if (is_own or pokemon.ability) else 0
        item_id = self.obs._item_id(pokemon) if (is_own or pokemon.item) else 0

        # Types
        types = np.array([1, 0], dtype=np.int8)
        if pokemon.types:
            if pokemon.types[0] is not None:
                types[0] = _TYPE_MAP.get(pokemon.types[0], 1)
            if len(pokemon.types) > 1 and pokemon.types[1] is not None:
                types[1] = _TYPE_MAP.get(pokemon.types[1], 0)

        # Base stats
        bst = getattr(pokemon, 'base_stats', {}) or {}
        stat_order = ['hp', 'atk', 'def', 'spa', 'spd', 'spe']
        base_stats = np.array([bst.get(s, 80) for s in stat_order], dtype=np.int16)

        # HP
        # In gen4randombattle the protocol broadcasts exact HP/maxHP for all
        # pokemon, so poke-env exposes current_hp and max_hp as integers even
        # for opponent pokemon.  Use them directly when available.
        if is_own:
            hp = np.int16(pokemon.current_hp or 0)
            max_hp = np.int16(pokemon.max_hp or max(int(pokemon.current_hp or 1), 1))
        else:
            exact_max = getattr(pokemon, 'max_hp', None)
            exact_cur = getattr(pokemon, 'current_hp', None)
            if exact_max and exact_max > 0:
                max_hp = np.int16(exact_max)
                hp = np.int16(max(int(exact_cur or 0), 0))
            else:
                # Fallback: estimate from base stats assuming 31 IVs, 85 EVs
                frac = pokemon.current_hp_fraction if pokemon.current_hp_fraction is not None else 1.0
                level = pokemon.level or 100
                est_hp = int((2 * base_stats[0] + 31 + 21) * level / 100 + level + 10)
                max_hp = np.int16(max(est_hp, 1))
                hp = np.int16(max(int(frac * est_hp), 0))

        # Boosts
        boosts_dict = dict(pokemon.boosts) if pokemon.boosts else {}
        boost_order = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion']
        boosts = np.array([boosts_dict.get(b, 0) for b in boost_order], dtype=np.int8)

        # Moves — delegate move ID lookup to ObsBridge
        move_ids = np.full(4, -1, dtype=np.int16)
        move_pp = np.zeros(4, dtype=np.int8)
        move_max_pp = np.zeros(4, dtype=np.int8)
        move_disabled = np.zeros(4, dtype=np.bool_)
        moves_list = list(pokemon.moves.values()) if pokemon.moves else []
        for i, m in enumerate(moves_list[:4]):
            move_ids[i] = self.obs._move_id(m)
            pp = m.current_pp if m.current_pp is not None else 1
            maxpp = m.max_pp if m.max_pp else max(pp, 1)
            move_pp[i] = min(pp, 127)
            move_max_pp[i] = min(maxpp, 127)
            if hasattr(m, 'is_disabled') and m.is_disabled:
                move_disabled[i] = True

        # Status
        status_code = _STATUS_MAP.get(pokemon.status, 0)
        sleep_turns = 0
        if status_code == 4:
            sleep_turns = min(getattr(pokemon, 'sleep_turns', 1) or 1, 7)

        # Volatiles bitmask
        vol_mask = np.uint32(0)
        vol_data = np.zeros(MAX_VOLATILES, dtype=np.int8)
        effects = getattr(pokemon, 'effects', {}) or {}
        for eff, turns in effects.items():
            bit = _EFFECT_TO_VOL_BIT.get(eff)
            if bit is not None:
                vol_mask |= np.uint32(1 << bit)
                if isinstance(turns, int):
                    vol_data[bit] = min(turns, 127)

        # Last move
        last_move_id = np.int16(-1)
        if hasattr(pokemon, 'last_move') and pokemon.last_move:
            last_move_id = np.int16(
                self.obs._find_id(pokemon.last_move.id, self.obs._move_lookup)
            )

        level = np.int8(pokemon.level or 100)
        weight_hg = np.int16(getattr(pokemon, 'weight', 50) * 10)

        # Nature — own pokemon's nature is known from the team sheet;
        # for opponents it is never revealed, so default to 0 (Hardy/neutral).
        nature_id = np.int8(0)
        if is_own:
            raw_nature = getattr(pokemon, 'nature', None)
            if raw_nature is not None:
                nature_key = _re.sub(r'[^a-z0-9]', '', str(raw_nature).lower())
                nature_id = np.int8(_NATURE_LOOKUP.get(nature_key, 0))

        return {
            'species_id': np.int16(species_id),
            'ability_id': np.int16(ability_id),
            'item_id': np.int16(item_id),
            'types': types, 'base_stats': base_stats,
            'hp': hp, 'max_hp': max_hp,
            'boosts': boosts,
            'move_ids': move_ids, 'move_pp': move_pp,
            'move_max_pp': move_max_pp, 'move_disabled': move_disabled,
            'status': np.int8(status_code), 'status_turns': np.int8(0),
            'sleep_turns': np.int8(sleep_turns),
            'volatiles': vol_mask, 'volatile_data': vol_data,
            'is_active': False, 'fainted': pokemon.fainted,
            'last_move_id': last_move_id,
            'level': level, 'gender': np.int8(0),
            'nature_id': nature_id, 'weight_hg': weight_hg,
        }

    # ── Side conditions ──────────────────────────────────────────────

    @staticmethod
    def _encode_side_conditions(sc_dict: dict) -> np.ndarray:
        buf = np.zeros(MAX_SIDE_CONDS, dtype=np.int8)
        for cond, val in (sc_dict or {}).items():
            idx = _SC_MAP.get(cond)
            if idx is not None:
                buf[idx] = min(int(val), 127)
        return buf

    # ── Field ────────────────────────────────────────────────────────

    @staticmethod
    def _encode_field(battle: AbstractBattle) -> FieldState:
        weather_code, weather_turns = np.int8(0), np.int8(0)
        for w, turns in (battle.weather or {}).items():
            weather_code = np.int8(_WEATHER_MAP.get(w, 0))
            weather_turns = np.int8(min(turns, 127))
            break

        trick_room, gravity = np.int8(0), np.int8(0)
        for f, turns in (battle.fields or {}).items():
            if f == Field.TRICK_ROOM:
                trick_room = np.int8(min(turns, 7))
            elif f == Field.GRAVITY:
                gravity = np.int8(min(turns, 7))

        return FieldState(
            weather=jnp.int8(weather_code),
            weather_turns=jnp.int8(weather_turns),
            weather_max_turns=jnp.int8(weather_turns),
            terrain=jnp.int8(0), terrain_turns=jnp.int8(0),
            trick_room=jnp.int8(trick_room),
            gravity=jnp.int8(gravity),
            magic_room=jnp.int8(0), wonder_room=jnp.int8(0),
        )

    # ── Main conversion ──────────────────────────────────────────────

    def battle_to_state(
        self,
        battle: AbstractBattle,
        rng_key: jnp.ndarray,
    ) -> tuple[BattleState, RevealState]:
        """Convert a live poke-env Battle → pokejax (BattleState, RevealState).

        Our side = player 0, opponent = player 1.
        Uses ObsBridge._get_stable_team_order for consistent slot assignment.
        """
        own_team = self.obs._get_stable_team_order(battle, is_own=True)
        opp_team = self.obs._get_stable_team_order(battle, is_own=False)
        own_active = battle.active_pokemon
        opp_active = battle.opponent_active_pokemon

        # Find active indices
        own_active_idx = 0
        for i, p in enumerate(own_team):
            if p is not None and p is own_active:
                own_active_idx = i
                break
        opp_active_idx = 0
        for i, p in enumerate(opp_team):
            if p is not None and p == opp_active:
                opp_active_idx = i
                break

        # Encode teams
        own_enc = [self._pokemon_to_engine(p, is_own=True) for p in own_team]
        opp_enc = [self._pokemon_to_engine(p, is_own=False) for p in opp_team]

        for i, e in enumerate(own_enc):
            e['is_active'] = (i == own_active_idx) and not e['fainted']
        for i, e in enumerate(opp_enc):
            e['is_active'] = (i == opp_active_idx) and not e['fainted']

        def _stack(own_list, opp_list, field):
            a = np.stack([e[field] for e in own_list])
            b = np.stack([e[field] for e in opp_list])
            return jnp.stack([jnp.asarray(a), jnp.asarray(b)], axis=0)

        own_sc = self._encode_side_conditions(battle.side_conditions)
        opp_sc = self._encode_side_conditions(battle.opponent_side_conditions)
        own_left = sum(1 for e in own_enc if not e['fainted'])
        opp_left = sum(1 for e in opp_enc if not e['fainted'])

        state = BattleState(
            sides_team_species_id=_stack(own_enc, opp_enc, 'species_id').astype(jnp.int16),
            sides_team_ability_id=_stack(own_enc, opp_enc, 'ability_id').astype(jnp.int16),
            sides_team_item_id=_stack(own_enc, opp_enc, 'item_id').astype(jnp.int16),
            sides_team_types=_stack(own_enc, opp_enc, 'types').astype(jnp.int8),
            sides_team_base_stats=_stack(own_enc, opp_enc, 'base_stats').astype(jnp.int16),
            sides_team_hp=_stack(own_enc, opp_enc, 'hp').astype(jnp.int16),
            sides_team_max_hp=_stack(own_enc, opp_enc, 'max_hp').astype(jnp.int16),
            sides_team_boosts=_stack(own_enc, opp_enc, 'boosts').astype(jnp.int8),
            sides_team_move_ids=_stack(own_enc, opp_enc, 'move_ids').astype(jnp.int16),
            sides_team_move_pp=_stack(own_enc, opp_enc, 'move_pp').astype(jnp.int8),
            sides_team_move_max_pp=_stack(own_enc, opp_enc, 'move_max_pp').astype(jnp.int8),
            sides_team_move_disabled=_stack(own_enc, opp_enc, 'move_disabled').astype(jnp.bool_),
            sides_team_status=_stack(own_enc, opp_enc, 'status').astype(jnp.int8),
            sides_team_status_turns=_stack(own_enc, opp_enc, 'status_turns').astype(jnp.int8),
            sides_team_sleep_turns=_stack(own_enc, opp_enc, 'sleep_turns').astype(jnp.int8),
            sides_team_volatiles=_stack(own_enc, opp_enc, 'volatiles').astype(jnp.uint32),
            sides_team_volatile_data=_stack(own_enc, opp_enc, 'volatile_data').astype(jnp.int8),
            sides_team_is_active=jnp.array([
                [own_enc[i]['is_active'] for i in range(6)],
                [opp_enc[i]['is_active'] for i in range(6)],
            ], dtype=jnp.bool_),
            sides_team_fainted=jnp.array([
                [own_enc[i]['fainted'] for i in range(6)],
                [opp_enc[i]['fainted'] for i in range(6)],
            ], dtype=jnp.bool_),
            sides_team_last_move_id=_stack(own_enc, opp_enc, 'last_move_id').astype(jnp.int16),
            sides_team_move_this_turn=jnp.zeros((2, 6), dtype=jnp.bool_),
            sides_team_times_attacked=jnp.zeros((2, 6), dtype=jnp.int8),
            sides_team_level=_stack(own_enc, opp_enc, 'level').astype(jnp.int8),
            sides_team_gender=_stack(own_enc, opp_enc, 'gender').astype(jnp.int8),
            sides_team_nature_id=_stack(own_enc, opp_enc, 'nature_id').astype(jnp.int8),
            sides_team_weight_hg=_stack(own_enc, opp_enc, 'weight_hg').astype(jnp.int16),
            sides_team_base_species_id=_stack(own_enc, opp_enc, 'species_id').astype(jnp.int16),
            sides_active_idx=jnp.array([own_active_idx, opp_active_idx], dtype=jnp.int8),
            sides_pokemon_left=jnp.array([own_left, opp_left], dtype=jnp.int8),
            sides_side_conditions=jnp.stack([
                jnp.asarray(own_sc), jnp.asarray(opp_sc),
            ], axis=0).astype(jnp.int8),
            sides_dynamax_turns=jnp.zeros(2, dtype=jnp.int8),
            sides_team_active_turns=jnp.zeros((2, 6), dtype=jnp.int8),
            sides_last_dmg_phys=jnp.zeros(2, dtype=jnp.int16),
            sides_last_dmg_spec=jnp.zeros(2, dtype=jnp.int16),
            field=self._encode_field(battle),
            turn=jnp.int16(battle.turn or 1),
            finished=jnp.bool_(False),
            winner=jnp.int8(-1),
            rng_key=rng_key,
        )

        # RevealState
        revealed_pokemon = jnp.zeros((2, 6), dtype=jnp.bool_)
        revealed_moves = jnp.zeros((2, 6, 4), dtype=jnp.bool_)
        revealed_ability = jnp.zeros((2, 6), dtype=jnp.bool_)
        revealed_item = jnp.zeros((2, 6), dtype=jnp.bool_)

        for i, p in enumerate(opp_team):
            if p is not None:
                revealed_pokemon = revealed_pokemon.at[0, i].set(True)
                if p.moves:
                    for j in range(min(len(list(p.moves)), 4)):
                        revealed_moves = revealed_moves.at[0, i, j].set(True)
                if p.ability:
                    revealed_ability = revealed_ability.at[0, i].set(True)
                if p.item:
                    revealed_item = revealed_item.at[0, i].set(True)

        for i, p in enumerate(own_team):
            if p is not None and p.moves:
                revealed_pokemon = revealed_pokemon.at[1, i].set(True)

        reveal = RevealState(
            revealed_moves=revealed_moves,
            revealed_pokemon=revealed_pokemon,
            revealed_ability=revealed_ability,
            revealed_item=revealed_item,
        )

        return state, reveal
