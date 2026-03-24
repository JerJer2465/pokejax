"""Tests for PokeTransformer and PokeMLP model architectures."""

import jax
import jax.numpy as jnp
import pytest

from pokejax.rl.model import (
    PokeTransformer, PokeMLP, create_model,
    N_TOKENS, N_ACTIONS, N_ATOMS,
)
from pokejax.rl.obs_builder import FLOAT_DIM_PER_POKEMON


# --- Fixtures ---

@pytest.fixture
def dummy_inputs():
    """Create dummy model inputs."""
    B = 4
    int_ids = jnp.zeros((B, N_TOKENS, 8), dtype=jnp.int32)
    float_feats = jnp.zeros((B, N_TOKENS, FLOAT_DIM_PER_POKEMON), dtype=jnp.float32)
    legal_mask = jnp.ones((B, N_ACTIONS), dtype=jnp.float32)
    return int_ids, float_feats, legal_mask


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


# --- Tests ---

class TestPokeTransformer:
    def test_output_shapes(self, key, dummy_inputs):
        model = PokeTransformer()
        int_ids, float_feats, legal_mask = dummy_inputs
        params = model.init(key, int_ids, float_feats, legal_mask)
        log_probs, value_probs, value = model.apply(params, int_ids, float_feats, legal_mask)

        B = int_ids.shape[0]
        assert log_probs.shape == (B, N_ACTIONS)
        assert value_probs.shape == (B, N_ATOMS)
        assert value.shape == (B,)

    def test_log_probs_sum_to_one(self, key, dummy_inputs):
        model = PokeTransformer()
        int_ids, float_feats, legal_mask = dummy_inputs
        params = model.init(key, int_ids, float_feats, legal_mask)
        log_probs, _, _ = model.apply(params, int_ids, float_feats, legal_mask)

        probs = jnp.exp(log_probs)
        assert jnp.allclose(probs.sum(-1), 1.0, atol=1e-5)

    def test_value_probs_sum_to_one(self, key, dummy_inputs):
        model = PokeTransformer()
        int_ids, float_feats, legal_mask = dummy_inputs
        params = model.init(key, int_ids, float_feats, legal_mask)
        _, value_probs, _ = model.apply(params, int_ids, float_feats, legal_mask)

        assert jnp.allclose(value_probs.sum(-1), 1.0, atol=1e-5)

    def test_legal_mask_respected(self, key, dummy_inputs):
        model = PokeTransformer()
        int_ids, float_feats, _ = dummy_inputs
        B = int_ids.shape[0]
        # Only action 0 is legal
        legal_mask = jnp.zeros((B, N_ACTIONS), dtype=jnp.float32).at[:, 0].set(1.0)
        params = model.init(key, int_ids, float_feats, legal_mask)
        log_probs, _, _ = model.apply(params, int_ids, float_feats, legal_mask)

        probs = jnp.exp(log_probs)
        assert jnp.allclose(probs[:, 0], 1.0, atol=1e-4)


class TestPokeMLP:
    def test_output_shapes(self, key, dummy_inputs):
        model = PokeMLP()
        int_ids, float_feats, legal_mask = dummy_inputs
        params = model.init(key, int_ids, float_feats, legal_mask)
        log_probs, value_probs, value = model.apply(params, int_ids, float_feats, legal_mask)

        B = int_ids.shape[0]
        assert log_probs.shape == (B, N_ACTIONS)
        assert value_probs.shape == (B, N_ATOMS)
        assert value.shape == (B,)

    def test_log_probs_sum_to_one(self, key, dummy_inputs):
        model = PokeMLP()
        int_ids, float_feats, legal_mask = dummy_inputs
        params = model.init(key, int_ids, float_feats, legal_mask)
        log_probs, _, _ = model.apply(params, int_ids, float_feats, legal_mask)

        probs = jnp.exp(log_probs)
        assert jnp.allclose(probs.sum(-1), 1.0, atol=1e-5)

    def test_value_probs_sum_to_one(self, key, dummy_inputs):
        model = PokeMLP()
        int_ids, float_feats, legal_mask = dummy_inputs
        params = model.init(key, int_ids, float_feats, legal_mask)
        _, value_probs, _ = model.apply(params, int_ids, float_feats, legal_mask)

        assert jnp.allclose(value_probs.sum(-1), 1.0, atol=1e-5)

    def test_legal_mask_respected(self, key, dummy_inputs):
        model = PokeMLP()
        int_ids, float_feats, _ = dummy_inputs
        B = int_ids.shape[0]
        legal_mask = jnp.zeros((B, N_ACTIONS), dtype=jnp.float32).at[:, 0].set(1.0)
        params = model.init(key, int_ids, float_feats, legal_mask)
        log_probs, _, _ = model.apply(params, int_ids, float_feats, legal_mask)

        probs = jnp.exp(log_probs)
        assert jnp.allclose(probs[:, 0], 1.0, atol=1e-4)

    def test_batch_size_one(self, key):
        model = PokeMLP()
        int_ids = jnp.zeros((1, N_TOKENS, 8), dtype=jnp.int32)
        float_feats = jnp.zeros((1, N_TOKENS, FLOAT_DIM_PER_POKEMON), dtype=jnp.float32)
        legal_mask = jnp.ones((1, N_ACTIONS), dtype=jnp.float32)
        params = model.init(key, int_ids, float_feats, legal_mask)
        log_probs, value_probs, value = model.apply(params, int_ids, float_feats, legal_mask)

        assert log_probs.shape == (1, N_ACTIONS)
        assert value_probs.shape == (1, N_ATOMS)
        assert value.shape == (1,)


class TestPokeMLP_CustomDims:
    """Test PokeMLP with non-default embedding and architecture dims."""

    def test_custom_architecture(self, key, dummy_inputs):
        model = PokeMLP(
            token_dim=64,
            hidden_dims=(256, 256),
            species_embed_dim=32,
            move_embed_dim=16,
        )
        int_ids, float_feats, legal_mask = dummy_inputs
        params = model.init(key, int_ids, float_feats, legal_mask)
        log_probs, value_probs, value = model.apply(
            params, int_ids, float_feats, legal_mask
        )
        B = int_ids.shape[0]
        assert log_probs.shape == (B, N_ACTIONS)
        assert value_probs.shape == (B, N_ATOMS)
        assert value.shape == (B,)

    def test_large_architecture(self, key, dummy_inputs):
        model = PokeMLP(
            token_dim=256,
            hidden_dims=(1024, 1024, 1024),
            species_embed_dim=128,
            move_embed_dim=64,
        )
        int_ids, float_feats, legal_mask = dummy_inputs
        params = model.init(key, int_ids, float_feats, legal_mask)
        log_probs, _, _ = model.apply(params, int_ids, float_feats, legal_mask)
        probs = jnp.exp(log_probs)
        assert jnp.allclose(probs.sum(-1), 1.0, atol=1e-5)

    def test_different_dims_different_params(self, key, dummy_inputs):
        """Different architecture configs produce different param counts."""
        int_ids, float_feats, legal_mask = dummy_inputs
        small = PokeMLP(token_dim=64, hidden_dims=(256, 256))
        large = PokeMLP(token_dim=256, hidden_dims=(1024, 1024, 512))
        p_small = small.init(key, int_ids, float_feats, legal_mask)
        p_large = large.init(key, int_ids, float_feats, legal_mask)
        n_small = sum(x.size for x in jax.tree.leaves(p_small))
        n_large = sum(x.size for x in jax.tree.leaves(p_large))
        assert n_large > n_small


class TestCreateModel:
    def test_create_transformer(self):
        model = create_model("transformer")
        assert isinstance(model, PokeTransformer)

    def test_create_mlp(self):
        model = create_model("mlp")
        assert isinstance(model, PokeMLP)

    def test_create_mlp_with_kwargs(self):
        model = create_model("mlp", token_dim=64, hidden_dims=(256, 256))
        assert isinstance(model, PokeMLP)
        assert model.token_dim == 64
        assert model.hidden_dims == (256, 256)

    def test_invalid_arch(self):
        with pytest.raises(ValueError, match="Unknown architecture"):
            create_model("rnn")

    def test_both_archs_same_interface(self, key, dummy_inputs):
        """Both architectures produce the same output shapes."""
        int_ids, float_feats, legal_mask = dummy_inputs

        for arch in ("transformer", "mlp"):
            model = create_model(arch)
            params = model.init(key, int_ids, float_feats, legal_mask)
            log_probs, value_probs, value = model.apply(
                params, int_ids, float_feats, legal_mask
            )

            B = int_ids.shape[0]
            assert log_probs.shape == (B, N_ACTIONS), f"{arch}: log_probs shape mismatch"
            assert value_probs.shape == (B, N_ATOMS), f"{arch}: value_probs shape mismatch"
            assert value.shape == (B,), f"{arch}: value shape mismatch"


class TestParamCount:
    def test_transformer_param_count(self, key, dummy_inputs):
        model = PokeTransformer()
        int_ids, float_feats, legal_mask = dummy_inputs
        params = model.init(key, int_ids, float_feats, legal_mask)
        n_params = sum(x.size for x in jax.tree.leaves(params))
        # Should be ~1.6M
        assert 1_000_000 < n_params < 3_000_000, f"Transformer has {n_params} params"

    def test_mlp_param_count(self, key, dummy_inputs):
        model = PokeMLP()
        int_ids, float_feats, legal_mask = dummy_inputs
        params = model.init(key, int_ids, float_feats, legal_mask)
        n_params = sum(x.size for x in jax.tree.leaves(params))
        # Should be ~2-4M
        assert 1_500_000 < n_params < 5_000_000, f"MLP has {n_params} params"
