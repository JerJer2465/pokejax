"""
L1 tests for Phase 4 move effects.

Tests:
  - move_effects_data: table builder produces correct rows
  - moves.py: execute_move_effects applies stat boosts, hazards, screens,
              weather, volatile bits correctly
  - hit_pipeline: self-heal from MF_HEAL_NUM/DEN
  - hit_pipeline: bypass type immunity for self/field moves
  - hit_pipeline: secondary stat change applied to foe
  - Full integration: stat-boost move raises attacker boosts in battle
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from pokejax.types import (
    BattleState,
    BOOST_ATK, BOOST_DEF, BOOST_SPA, BOOST_SPD, BOOST_SPE, BOOST_ACC, BOOST_EVA,
    SC_SPIKES, SC_STEALTHROCK, SC_REFLECT, SC_SAFEGUARD,
    WEATHER_SUN, WEATHER_RAIN,
    VOL_PROTECT, VOL_FOCUSENERGY, VOL_SEEDED,
)
from pokejax.data.move_effects_data import (
    build_move_effects_table,
    ME_SELF_BOOST, ME_FOE_LOWER, ME_HAZARD, ME_SCREEN,
    ME_WEATHER, ME_TRICK_ROOM, ME_VOLATILE_SELF, ME_VOLATILE_FOE,
    GEN4_MOVE_EFFECTS, NONE_STAT, MOVE_EFFECT_FIELDS,
)
from pokejax.mechanics.moves import execute_move_effects
from pokejax.data.tables import load_tables
from pokejax.config import GenConfig
from pokejax.core.state import make_battle_state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tables():
    return load_tables(gen=4)


@pytest.fixture(scope="module")
def cfg():
    return GenConfig.for_gen(4)


def _make_state(tables, cfg):
    """Minimal BattleState for effect tests."""
    n = 6
    base = np.array([
        [100, 100, 100, 100, 100, 100],
    ] * n, dtype=np.int16)
    hp = np.full(n, 300, dtype=np.int16)
    types = np.zeros((n, 2), dtype=np.int8)
    types[:, 0] = 1  # Normal
    move_ids = np.tile(np.arange(4, dtype=np.int16), (n, 1))
    move_pp = np.full((n, 4), 35, dtype=np.int8)
    move_max_pp = np.full((n, 4), 35, dtype=np.int8)
    species = np.zeros(n, dtype=np.int16)
    abilities = np.zeros(n, dtype=np.int16)
    items = np.zeros(n, dtype=np.int16)
    levels = np.full(n, 100, dtype=np.int8)
    genders = np.zeros(n, dtype=np.int8)
    natures = np.zeros(n, dtype=np.int8)
    weights = np.full(n, 100, dtype=np.int16)

    return make_battle_state(
        p1_species=species, p2_species=species,
        p1_abilities=abilities, p2_abilities=abilities,
        p1_items=items, p2_items=items,
        p1_types=types, p2_types=types,
        p1_base_stats=base, p2_base_stats=base,
        p1_max_hp=hp, p2_max_hp=hp,
        p1_move_ids=move_ids, p2_move_ids=move_ids,
        p1_move_pp=move_pp, p2_move_pp=move_pp,
        p1_move_max_pp=move_max_pp, p2_move_max_pp=move_max_pp,
        p1_levels=levels, p2_levels=levels,
        p1_genders=genders, p2_genders=genders,
        p1_natures=natures, p2_natures=natures,
        p1_weights_hg=weights, p2_weights_hg=weights,
        rng_key=jax.random.PRNGKey(0),
    )


# ---------------------------------------------------------------------------
# move_effects_data tests
# ---------------------------------------------------------------------------

class TestMoveEffectsData:
    def test_swords_dance_in_dict(self):
        effect = GEN4_MOVE_EFFECTS["Swords Dance"]
        assert effect[0] == ME_SELF_BOOST
        assert effect[1] == BOOST_ATK
        assert effect[2] == 2  # +2 ATK

    def test_calm_mind_two_boosts(self):
        effect = GEN4_MOVE_EFFECTS["Calm Mind"]
        assert effect[0] == ME_SELF_BOOST
        assert effect[1] == BOOST_SPA
        assert effect[2] == 1
        assert effect[3] == BOOST_SPD
        assert effect[4] == 1

    def test_dragon_dance_two_boosts(self):
        effect = GEN4_MOVE_EFFECTS["Dragon Dance"]
        assert effect[0] == ME_SELF_BOOST
        assert effect[1] == BOOST_ATK
        assert effect[3] == BOOST_SPE

    def test_curse_three_boosts(self):
        effect = GEN4_MOVE_EFFECTS["Curse"]
        assert effect[0] == ME_SELF_BOOST
        assert effect[5] == BOOST_SPE
        assert effect[6] == -1  # -1 SPE

    def test_growl_foe_lower(self):
        effect = GEN4_MOVE_EFFECTS["Growl"]
        assert effect[0] == ME_FOE_LOWER
        assert effect[1] == BOOST_ATK
        assert effect[2] == -1

    def test_spikes_hazard(self):
        effect = GEN4_MOVE_EFFECTS["Spikes"]
        assert effect[0] == ME_HAZARD
        assert effect[1] == SC_SPIKES
        assert effect[2] == 3  # max 3 layers

    def test_stealth_rock_hazard(self):
        effect = GEN4_MOVE_EFFECTS["Stealth Rock"]
        assert effect[0] == ME_HAZARD
        assert effect[1] == SC_STEALTHROCK

    def test_reflect_screen(self):
        effect = GEN4_MOVE_EFFECTS["Reflect"]
        assert effect[0] == ME_SCREEN
        assert effect[1] == SC_REFLECT
        assert effect[2] == 5  # 5 turns

    def test_sunny_day_weather(self):
        effect = GEN4_MOVE_EFFECTS["Sunny Day"]
        assert effect[0] == ME_WEATHER
        assert effect[1] == WEATHER_SUN

    def test_trick_room(self):
        effect = GEN4_MOVE_EFFECTS["Trick Room"]
        assert effect[0] == ME_TRICK_ROOM

    def test_protect_volatile_self(self):
        effect = GEN4_MOVE_EFFECTS["Protect"]
        assert effect[0] == ME_VOLATILE_SELF
        assert effect[1] == VOL_PROTECT

    def test_leech_seed_volatile_foe(self):
        effect = GEN4_MOVE_EFFECTS["Leech Seed"]
        assert effect[0] == ME_VOLATILE_FOE
        assert effect[1] == VOL_SEEDED

    def test_build_table_correct_rows(self):
        """build_move_effects_table maps names to the right rows."""
        name_to_id = {"Swords Dance": 5, "Growl": 12, "Sunny Day": 99}
        table = build_move_effects_table(name_to_id, n_moves=200)
        assert table.shape == (200, MOVE_EFFECT_FIELDS)
        # Swords Dance row
        assert table[5, 0] == ME_SELF_BOOST
        assert table[5, 1] == BOOST_ATK
        assert table[5, 2] == 2
        # Growl row
        assert table[12, 0] == ME_FOE_LOWER
        # Sunny Day row
        assert table[99, 0] == ME_WEATHER
        assert table[99, 1] == WEATHER_SUN
        # Unknown move row stays zeroed
        assert table[0, 0] == 0

    def test_none_stat_sentinels(self):
        """Unused stat slots start as NONE_STAT."""
        name_to_id = {"Agility": 7}
        table = build_move_effects_table(name_to_id, n_moves=10)
        # Agility only sets stat1; stat2 and stat3 should be NONE_STAT
        assert table[7, 3] == NONE_STAT  # stat2
        assert table[7, 5] == NONE_STAT  # stat3


# ---------------------------------------------------------------------------
# execute_move_effects tests
# ---------------------------------------------------------------------------

class TestExecuteMoveEffects:
    """Tests drive execute_move_effects directly by crafting a move_effects table."""

    def _make_tables_with_effect(self, tables_real, effect_tuple, move_idx=0):
        """Return a Tables-like object with a custom move_effects row."""
        import dataclasses
        n = tables_real.move_effects.shape[0]
        eff_np = np.zeros((n, MOVE_EFFECT_FIELDS), dtype=np.int16)
        eff_np[:, 1] = NONE_STAT
        eff_np[:, 3] = NONE_STAT
        eff_np[:, 5] = NONE_STAT
        if move_idx < n:
            eff_np[move_idx] = list(effect_tuple)
        custom_eff = jnp.array(eff_np)
        # Return a dataclass copy with replaced move_effects
        return dataclasses.replace(tables_real, move_effects=custom_eff)

    def test_self_boost_atk(self, tables, cfg):
        """ME_SELF_BOOST raises attacker's ATK boost by the amount."""
        state = _make_state(tables, cfg)
        effect = GEN4_MOVE_EFFECTS["Swords Dance"]  # +2 ATK
        t = self._make_tables_with_effect(tables, effect, move_idx=0)
        move_id = jnp.int16(0)
        key = jax.random.PRNGKey(1)
        atk_idx = int(state.sides_active_idx[0])

        before = int(state.sides_team_boosts[0, atk_idx, BOOST_ATK])
        new_state, _ = execute_move_effects(
            t, state, atk_side=0, def_side=1,
            move_id=move_id, cancelled=jnp.bool_(False),
            key=key, cfg=cfg,
        )
        after = int(new_state.sides_team_boosts[0, atk_idx, BOOST_ATK])
        assert after == before + 2

    def test_self_boost_cancelled_noop(self, tables, cfg):
        """Effect is suppressed when cancelled=True."""
        state = _make_state(tables, cfg)
        effect = GEN4_MOVE_EFFECTS["Swords Dance"]
        t = self._make_tables_with_effect(tables, effect, move_idx=0)
        move_id = jnp.int16(0)
        key = jax.random.PRNGKey(2)
        atk_idx = int(state.sides_active_idx[0])

        before = int(state.sides_team_boosts[0, atk_idx, BOOST_ATK])
        new_state, _ = execute_move_effects(
            t, state, atk_side=0, def_side=1,
            move_id=move_id, cancelled=jnp.bool_(True),
            key=key, cfg=cfg,
        )
        after = int(new_state.sides_team_boosts[0, atk_idx, BOOST_ATK])
        assert after == before  # no change

    def test_calm_mind_two_stats(self, tables, cfg):
        """ME_SELF_BOOST with two stat/amount pairs raises both stats."""
        state = _make_state(tables, cfg)
        effect = GEN4_MOVE_EFFECTS["Calm Mind"]  # +1 SPA, +1 SPD
        t = self._make_tables_with_effect(tables, effect, move_idx=1)
        move_id = jnp.int16(1)
        key = jax.random.PRNGKey(3)
        atk_idx = int(state.sides_active_idx[0])

        new_state, _ = execute_move_effects(
            t, state, atk_side=0, def_side=1,
            move_id=move_id, cancelled=jnp.bool_(False),
            key=key, cfg=cfg,
        )
        assert int(new_state.sides_team_boosts[0, atk_idx, BOOST_SPA]) == 1
        assert int(new_state.sides_team_boosts[0, atk_idx, BOOST_SPD]) == 1

    def test_curse_three_stats(self, tables, cfg):
        """Curse raises ATK+DEF and lowers SPE."""
        state = _make_state(tables, cfg)
        effect = GEN4_MOVE_EFFECTS["Curse"]
        t = self._make_tables_with_effect(tables, effect, move_idx=0)
        move_id = jnp.int16(0)
        key = jax.random.PRNGKey(4)
        atk_idx = int(state.sides_active_idx[0])

        new_state, _ = execute_move_effects(
            t, state, atk_side=0, def_side=1,
            move_id=move_id, cancelled=jnp.bool_(False),
            key=key, cfg=cfg,
        )
        assert int(new_state.sides_team_boosts[0, atk_idx, BOOST_ATK]) == 1
        assert int(new_state.sides_team_boosts[0, atk_idx, BOOST_DEF]) == 1
        assert int(new_state.sides_team_boosts[0, atk_idx, BOOST_SPE]) == -1

    def test_boost_clamps_at_plus_six(self, tables, cfg):
        """Boost stage is clamped to +6."""
        state = _make_state(tables, cfg)
        # Manually set ATK boost to +5
        atk_idx = int(state.sides_active_idx[0])
        boosts = state.sides_team_boosts.at[0, atk_idx, BOOST_ATK].set(jnp.int8(5))
        state = state._replace(sides_team_boosts=boosts)

        effect = GEN4_MOVE_EFFECTS["Swords Dance"]  # +2 ATK (would overshoot)
        t = self._make_tables_with_effect(tables, effect, move_idx=0)
        new_state, _ = execute_move_effects(
            t, state, atk_side=0, def_side=1,
            move_id=jnp.int16(0), cancelled=jnp.bool_(False),
            key=jax.random.PRNGKey(5), cfg=cfg,
        )
        assert int(new_state.sides_team_boosts[0, atk_idx, BOOST_ATK]) == 6

    def test_foe_lower(self, tables, cfg):
        """ME_FOE_LOWER applies to defender's stats."""
        state = _make_state(tables, cfg)
        effect = GEN4_MOVE_EFFECTS["Growl"]  # -1 ATK on foe
        t = self._make_tables_with_effect(tables, effect, move_idx=0)
        move_id = jnp.int16(0)
        key = jax.random.PRNGKey(6)
        def_idx = int(state.sides_active_idx[1])

        before = int(state.sides_team_boosts[1, def_idx, BOOST_ATK])
        new_state, _ = execute_move_effects(
            t, state, atk_side=0, def_side=1,
            move_id=move_id, cancelled=jnp.bool_(False),
            key=key, cfg=cfg,
        )
        after = int(new_state.sides_team_boosts[1, def_idx, BOOST_ATK])
        assert after == before - 1

    def test_hazard_adds_spikes_layer(self, tables, cfg):
        """ME_HAZARD increments side condition layer count."""
        state = _make_state(tables, cfg)
        effect = GEN4_MOVE_EFFECTS["Spikes"]
        t = self._make_tables_with_effect(tables, effect, move_idx=0)

        before = int(state.sides_side_conditions[1, SC_SPIKES])
        new_state, _ = execute_move_effects(
            t, state, atk_side=0, def_side=1,
            move_id=jnp.int16(0), cancelled=jnp.bool_(False),
            key=jax.random.PRNGKey(7), cfg=cfg,
        )
        assert int(new_state.sides_side_conditions[1, SC_SPIKES]) == before + 1

    def test_hazard_capped_at_max_layers(self, tables, cfg):
        """Spikes capped at 3 layers."""
        state = _make_state(tables, cfg)
        # Pre-set to 3 layers
        sc = state.sides_side_conditions.at[1, SC_SPIKES].set(jnp.int8(3))
        state = state._replace(sides_side_conditions=sc)

        effect = GEN4_MOVE_EFFECTS["Spikes"]
        t = self._make_tables_with_effect(tables, effect, move_idx=0)

        new_state, _ = execute_move_effects(
            t, state, atk_side=0, def_side=1,
            move_id=jnp.int16(0), cancelled=jnp.bool_(False),
            key=jax.random.PRNGKey(8), cfg=cfg,
        )
        assert int(new_state.sides_side_conditions[1, SC_SPIKES]) == 3  # unchanged

    def test_screen_sets_turns(self, tables, cfg):
        """ME_SCREEN sets the screen turn counter on attacker's side."""
        state = _make_state(tables, cfg)
        effect = GEN4_MOVE_EFFECTS["Reflect"]
        t = self._make_tables_with_effect(tables, effect, move_idx=0)

        new_state, _ = execute_move_effects(
            t, state, atk_side=0, def_side=1,
            move_id=jnp.int16(0), cancelled=jnp.bool_(False),
            key=jax.random.PRNGKey(9), cfg=cfg,
        )
        assert int(new_state.sides_side_conditions[0, SC_REFLECT]) == 5

    def test_screen_no_overwrite_if_active(self, tables, cfg):
        """Screen does not reset turn count if already active."""
        state = _make_state(tables, cfg)
        sc = state.sides_side_conditions.at[0, SC_REFLECT].set(jnp.int8(3))
        state = state._replace(sides_side_conditions=sc)

        effect = GEN4_MOVE_EFFECTS["Reflect"]
        t = self._make_tables_with_effect(tables, effect, move_idx=0)

        new_state, _ = execute_move_effects(
            t, state, atk_side=0, def_side=1,
            move_id=jnp.int16(0), cancelled=jnp.bool_(False),
            key=jax.random.PRNGKey(10), cfg=cfg,
        )
        # Should keep remaining turns, not reset to 5
        assert int(new_state.sides_side_conditions[0, SC_REFLECT]) == 3

    def test_weather_sets_sun(self, tables, cfg):
        """ME_WEATHER correctly sets weather and turn count."""
        state = _make_state(tables, cfg)
        effect = GEN4_MOVE_EFFECTS["Sunny Day"]
        t = self._make_tables_with_effect(tables, effect, move_idx=0)

        new_state, _ = execute_move_effects(
            t, state, atk_side=0, def_side=1,
            move_id=jnp.int16(0), cancelled=jnp.bool_(False),
            key=jax.random.PRNGKey(11), cfg=cfg,
        )
        assert int(new_state.field.weather) == WEATHER_SUN
        assert int(new_state.field.weather_turns) == 5

    def test_weather_cancelled_noop(self, tables, cfg):
        """Cancelled weather move leaves field unchanged."""
        state = _make_state(tables, cfg)
        effect = GEN4_MOVE_EFFECTS["Rain Dance"]
        t = self._make_tables_with_effect(tables, effect, move_idx=0)

        new_state, _ = execute_move_effects(
            t, state, atk_side=0, def_side=1,
            move_id=jnp.int16(0), cancelled=jnp.bool_(True),
            key=jax.random.PRNGKey(12), cfg=cfg,
        )
        assert int(new_state.field.weather) == int(state.field.weather)

    def test_trick_room_toggles_on(self, tables, cfg):
        """Trick Room sets 5 turns when not active."""
        state = _make_state(tables, cfg)
        effect = GEN4_MOVE_EFFECTS["Trick Room"]
        t = self._make_tables_with_effect(tables, effect, move_idx=0)

        new_state, _ = execute_move_effects(
            t, state, atk_side=0, def_side=1,
            move_id=jnp.int16(0), cancelled=jnp.bool_(False),
            key=jax.random.PRNGKey(13), cfg=cfg,
        )
        assert int(new_state.field.trick_room) == 5

    def test_trick_room_toggles_off(self, tables, cfg):
        """Trick Room clears to 0 when already active."""
        state = _make_state(tables, cfg)
        new_field = state.field._replace(trick_room=jnp.int8(3))
        state = state._replace(field=new_field)

        effect = GEN4_MOVE_EFFECTS["Trick Room"]
        t = self._make_tables_with_effect(tables, effect, move_idx=0)

        new_state, _ = execute_move_effects(
            t, state, atk_side=0, def_side=1,
            move_id=jnp.int16(0), cancelled=jnp.bool_(False),
            key=jax.random.PRNGKey(14), cfg=cfg,
        )
        assert int(new_state.field.trick_room) == 0

    def test_volatile_self(self, tables, cfg):
        """ME_VOLATILE_SELF sets bit on attacker."""
        state = _make_state(tables, cfg)
        effect = GEN4_MOVE_EFFECTS["Focus Energy"]
        t = self._make_tables_with_effect(tables, effect, move_idx=0)
        atk_idx = int(state.sides_active_idx[0])

        new_state, _ = execute_move_effects(
            t, state, atk_side=0, def_side=1,
            move_id=jnp.int16(0), cancelled=jnp.bool_(False),
            key=jax.random.PRNGKey(15), cfg=cfg,
        )
        vol = int(new_state.sides_team_volatiles[0, atk_idx])
        assert vol & (1 << VOL_FOCUSENERGY)

    def test_volatile_foe(self, tables, cfg):
        """ME_VOLATILE_FOE sets bit on defender."""
        state = _make_state(tables, cfg)
        effect = GEN4_MOVE_EFFECTS["Leech Seed"]
        t = self._make_tables_with_effect(tables, effect, move_idx=0)
        def_idx = int(state.sides_active_idx[1])

        new_state, _ = execute_move_effects(
            t, state, atk_side=0, def_side=1,
            move_id=jnp.int16(0), cancelled=jnp.bool_(False),
            key=jax.random.PRNGKey(16), cfg=cfg,
        )
        vol = int(new_state.sides_team_volatiles[1, def_idx])
        assert vol & (1 << VOL_SEEDED)

    def test_no_effect_noop(self, tables, cfg):
        """ME_NONE leaves state completely unchanged."""
        state = _make_state(tables, cfg)
        # Use move index 0 in the real tables (which has ME_NONE by default for Tackle)
        move_id = jnp.int16(0)
        key = jax.random.PRNGKey(17)

        new_state, _ = execute_move_effects(
            tables, state, atk_side=0, def_side=1,
            move_id=move_id, cancelled=jnp.bool_(False),
            key=key, cfg=cfg,
        )
        # Weather, terrain, trick_room, side conditions, boosts all unchanged
        assert int(new_state.field.weather) == int(state.field.weather)
        assert int(new_state.field.trick_room) == int(state.field.trick_room)
        atk_idx = int(state.sides_active_idx[0])
        for b in range(7):
            assert int(new_state.sides_team_boosts[0, atk_idx, b]) == \
                   int(state.sides_team_boosts[0, atk_idx, b])


# ---------------------------------------------------------------------------
# hit_pipeline tests (type immunity bypass + healing)
# ---------------------------------------------------------------------------

class TestHitPipelinePhase4:
    def test_self_targeting_move_not_cancelled_by_ghost(self, tables, cfg):
        """
        Self-targeting status moves (target=1) bypass type immunity.
        Swords Dance (Normal-type) should not be cancelled vs Ghost defender.
        """
        from pokejax.engine.hit_pipeline import execute_move_hit
        import dataclasses

        state = _make_state(tables, cfg)
        # Set defender to Ghost-type so Normal type would be immune
        from pokejax.types import TYPE_GHOST
        def_idx = int(state.sides_active_idx[1])
        types_arr = state.sides_team_types.at[1, def_idx, 0].set(jnp.int8(TYPE_GHOST))
        state = state._replace(sides_team_types=types_arr)

        # Build a custom move table: move 0 = Normal-type, Status, target=self(1), bp=0
        from pokejax.data.extractor import MOVE_FIELDS
        moves_np = np.zeros((4, MOVE_FIELDS), dtype=np.int16)
        moves_np[0] = [0, 101, 1, 2, 0, 10, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #              bp acc  ty cat pri pp  tgt ...  (type=1=Normal, category=2=Status, target=1=self)

        from pokejax.data.move_effects_data import MOVE_EFFECT_FIELDS, NONE_STAT
        eff_np = np.zeros((4, MOVE_EFFECT_FIELDS), dtype=np.int16)
        eff_np[:, 1] = NONE_STAT
        eff_np[:, 3] = NONE_STAT
        eff_np[:, 5] = NONE_STAT

        t = dataclasses.replace(
            tables,
            moves=jnp.array(moves_np),
            move_effects=jnp.array(eff_np),
        )

        key = jax.random.PRNGKey(20)
        # Should NOT raise or produce cancelled=True
        state_out, dmg, crit, key_out, cancelled = execute_move_hit(
            t, state, atk_side=0, def_side=1,
            move_id=jnp.int16(0), key=key, cfg=cfg,
        )
        # cancelled should be False (self-targeting bypasses ghost immunity)
        assert not bool(cancelled), "Self-targeting move should not be type-cancelled"

    def test_field_targeting_not_cancelled(self, tables, cfg):
        """
        Field-targeting moves (target=foeSide/allySide) bypass type immunity.
        Spikes (Ground-type) vs Flying-type should NOT be cancelled.
        """
        from pokejax.engine.hit_pipeline import execute_move_hit
        import dataclasses
        from pokejax.types import TYPE_FLYING

        state = _make_state(tables, cfg)
        # Set defender to Flying-type (immune to Ground)
        def_idx = int(state.sides_active_idx[1])
        types_arr = state.sides_team_types.at[1, def_idx, 0].set(jnp.int8(TYPE_FLYING))
        state = state._replace(sides_team_types=types_arr)

        from pokejax.data.extractor import MOVE_FIELDS
        moves_np = np.zeros((4, MOVE_FIELDS), dtype=np.int16)
        # Spikes: Ground-type, Status, target=foeSide(10), always hits (101)
        moves_np[0] = [0, 101, 9, 2, 0, 20, 10, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #              bp acc  ty cat pri pp  tgt ... (type=9=Ground, target=10=foeSide)

        from pokejax.data.move_effects_data import MOVE_EFFECT_FIELDS, NONE_STAT
        eff_np = np.zeros((4, MOVE_EFFECT_FIELDS), dtype=np.int16)
        eff_np[:, 1] = NONE_STAT; eff_np[:, 3] = NONE_STAT; eff_np[:, 5] = NONE_STAT

        t = dataclasses.replace(
            tables,
            moves=jnp.array(moves_np),
            move_effects=jnp.array(eff_np),
        )

        key = jax.random.PRNGKey(21)
        _, dmg, crit, key_out, cancelled = execute_move_hit(
            t, state, atk_side=0, def_side=1,
            move_id=jnp.int16(0), key=key, cfg=cfg,
        )
        assert not bool(cancelled), "Field move vs Flying should NOT be type-cancelled"

    def test_self_heal_move(self, tables, cfg):
        """
        Move with MF_HEAL_NUM=1, MF_HEAL_DEN=2 heals attacker by 50% max HP.
        """
        from pokejax.engine.hit_pipeline import execute_move_hit
        import dataclasses

        state = _make_state(tables, cfg)
        # Reduce attacker HP to 50 to verify healing
        atk_idx = int(state.sides_active_idx[0])
        hp_arr = state.sides_team_hp.at[0, atk_idx].set(jnp.int16(50))
        state = state._replace(sides_team_hp=hp_arr)

        from pokejax.data.extractor import MOVE_FIELDS
        moves_np = np.zeros((4, MOVE_FIELDS), dtype=np.int16)
        # Recover: 0 bp, always hits (101), Normal, Status, target=self(1), heal 1/2
        # heal_num=1 at index 20, heal_den=2 at index 21
        moves_np[0] = [0, 101, 1, 2, 0, 10, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2]

        from pokejax.data.move_effects_data import MOVE_EFFECT_FIELDS, NONE_STAT
        eff_np = np.zeros((4, MOVE_EFFECT_FIELDS), dtype=np.int16)
        eff_np[:, 1] = NONE_STAT; eff_np[:, 3] = NONE_STAT; eff_np[:, 5] = NONE_STAT

        t = dataclasses.replace(
            tables,
            moves=jnp.array(moves_np),
            move_effects=jnp.array(eff_np),
        )

        key = jax.random.PRNGKey(22)
        state_out, dmg, crit, key_out, cancelled = execute_move_hit(
            t, state, atk_side=0, def_side=1,
            move_id=jnp.int16(0), key=key, cfg=cfg,
        )
        max_hp = int(state.sides_team_max_hp[0, atk_idx])
        hp_after = int(state_out.sides_team_hp[0, atk_idx])
        # Should have healed by 50% of 300 = 150, capped at max_hp
        expected = min(50 + max_hp // 2, max_hp)
        assert hp_after == expected, f"Expected {expected} HP, got {hp_after}"

    def test_secondary_stat_change(self, tables, cfg):
        """
        Move with MF_SEC_BOOST_STAT and MF_SEC_BOOST_AMT applies stat change to foe
        at the secondary chance.
        """
        from pokejax.engine.hit_pipeline import execute_move_hit
        import dataclasses

        from pokejax.data.extractor import MOVE_FIELDS
        moves_np = np.zeros((4, MOVE_FIELDS), dtype=np.int16)
        # Fake Psychic: Special, 90 bp, Psychic-type, 100% secondary, -1 SPD on foe
        # MF_SEC_CHANCE=100, MF_SEC_STATUS=0, MF_SEC_BOOST_STAT=3(SPD), MF_SEC_BOOST_AMT=-1
        moves_np[0] = [90, 100, 11, 1, 0, 10, 0, 0, 0, 1, 100, 0, BOOST_SPD, -1, 0, 0, 0, 0, 0, 0, 0, 0]

        from pokejax.data.move_effects_data import MOVE_EFFECT_FIELDS, NONE_STAT
        eff_np = np.zeros((4, MOVE_EFFECT_FIELDS), dtype=np.int16)
        eff_np[:, 1] = NONE_STAT; eff_np[:, 3] = NONE_STAT; eff_np[:, 5] = NONE_STAT

        t = dataclasses.replace(
            tables,
            moves=jnp.array(moves_np),
            move_effects=jnp.array(eff_np),
        )

        state = _make_state(tables, cfg)
        def_idx = int(state.sides_active_idx[1])
        before_spd = int(state.sides_team_boosts[1, def_idx, BOOST_SPD])

        key = jax.random.PRNGKey(23)
        state_out, dmg, crit, key_out, cancelled = execute_move_hit(
            t, state, atk_side=0, def_side=1,
            move_id=jnp.int16(0), key=key, cfg=cfg,
        )
        after_spd = int(state_out.sides_team_boosts[1, def_idx, BOOST_SPD])
        # 100% chance so must have triggered
        assert after_spd == before_spd - 1, \
            f"Expected SPD boost {before_spd-1}, got {after_spd}"
