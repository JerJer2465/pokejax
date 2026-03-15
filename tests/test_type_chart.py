"""L1 property tests: type effectiveness."""

import jax.numpy as jnp
import pytest

from pokejax.types import (
    TYPE_NORMAL, TYPE_FIRE, TYPE_WATER, TYPE_ELECTRIC, TYPE_GRASS,
    TYPE_ICE, TYPE_FIGHTING, TYPE_POISON, TYPE_GROUND, TYPE_FLYING,
    TYPE_PSYCHIC, TYPE_BUG, TYPE_ROCK, TYPE_GHOST, TYPE_DRAGON,
    TYPE_DARK, TYPE_STEEL, TYPE_FAIRY, TYPE_NONE,
)
from pokejax.core.damage import type_effectiveness


class TestTypeChart:
    def test_neutral(self, tables4):
        eff = type_effectiveness(tables4, TYPE_NORMAL, TYPE_NORMAL, TYPE_NONE)
        assert float(eff) == pytest.approx(1.0)

    def test_super_effective_fire_grass(self, tables4):
        eff = type_effectiveness(tables4, TYPE_FIRE, TYPE_GRASS, TYPE_NONE)
        assert float(eff) == pytest.approx(2.0)

    def test_super_effective_water_fire(self, tables4):
        eff = type_effectiveness(tables4, TYPE_WATER, TYPE_FIRE, TYPE_NONE)
        assert float(eff) == pytest.approx(2.0)

    def test_not_very_effective_fire_water(self, tables4):
        eff = type_effectiveness(tables4, TYPE_FIRE, TYPE_WATER, TYPE_NONE)
        assert float(eff) == pytest.approx(0.5)

    def test_immune_normal_ghost(self, tables4):
        eff = type_effectiveness(tables4, TYPE_NORMAL, TYPE_GHOST, TYPE_NONE)
        assert float(eff) == pytest.approx(0.0)

    def test_immune_electric_ground(self, tables4):
        eff = type_effectiveness(tables4, TYPE_ELECTRIC, TYPE_GROUND, TYPE_NONE)
        assert float(eff) == pytest.approx(0.0)

    def test_immune_ground_flying(self, tables4):
        eff = type_effectiveness(tables4, TYPE_GROUND, TYPE_FLYING, TYPE_NONE)
        assert float(eff) == pytest.approx(0.0)

    def test_4x_effective_ice_flying_grass(self, tables4):
        # Ice vs Grass/Flying = 2x * 2x = 4x
        eff = type_effectiveness(tables4, TYPE_ICE, TYPE_GRASS, TYPE_FLYING)
        assert float(eff) == pytest.approx(4.0)

    def test_025x_effective_fighting_ghost_flying(self, tables4):
        # Fighting vs Ghost/Flying: Ghost immune (0x) * 1x = 0x
        eff = type_effectiveness(tables4, TYPE_FIGHTING, TYPE_GHOST, TYPE_FLYING)
        assert float(eff) == pytest.approx(0.0)

    def test_dual_type_nve(self, tables4):
        # Water vs Grass/Dragon: 0.5x * 0.5x = 0.25x
        eff = type_effectiveness(tables4, TYPE_WATER, TYPE_GRASS, TYPE_DRAGON)
        assert float(eff) == pytest.approx(0.25)

    def test_single_type_sentinel(self, tables4):
        # sentinel type2=0 should act as 1.0x multiplier
        eff_dual   = type_effectiveness(tables4, TYPE_FIRE, TYPE_GRASS, TYPE_NONE)
        eff_single = type_effectiveness(tables4, TYPE_FIRE, TYPE_GRASS, jnp.int32(0))
        assert float(eff_dual) == float(eff_single)

    def test_gen4_no_fairy(self, tables4):
        # In Gen 4 tables, Fairy type should be neutral everywhere
        eff = type_effectiveness(tables4, TYPE_FAIRY, TYPE_DRAGON, TYPE_NONE)
        # Gen 4 doesn't have Fairy, so it's neutral (1.0), not super effective
        assert float(eff) == pytest.approx(1.0)

    def test_ghost_immune_to_normal_and_fighting(self, tables4):
        assert float(type_effectiveness(tables4, TYPE_NORMAL, TYPE_GHOST, TYPE_NONE)) == 0.0
        assert float(type_effectiveness(tables4, TYPE_FIGHTING, TYPE_GHOST, TYPE_NONE)) == 0.0

    def test_psychic_immune_dark(self, tables4):
        eff = type_effectiveness(tables4, TYPE_PSYCHIC, TYPE_DARK, TYPE_NONE)
        assert float(eff) == pytest.approx(0.0)

    def test_poison_immune_vs_steel(self, tables4):
        # Gen 2-5: Poison-type moves do 0x to Steel (immune in type chart)
        eff = type_effectiveness(tables4, TYPE_POISON, TYPE_STEEL, TYPE_NONE)
        assert float(eff) == pytest.approx(0.0)
