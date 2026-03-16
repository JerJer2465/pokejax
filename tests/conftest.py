"""
Shared fixtures for PokeJAX tests.
"""

import pytest
import jax

# Persistent XLA compilation cache — first run compiles once; subsequent runs
# are instant.  Place this before any JAX import side effects.
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_compile_cache")

from pokejax.data.tables import load_tables
from pokejax.config import GenConfig


def pytest_addoption(parser):
    """Add custom CLI options for differential tests."""
    parser.addoption("--battles", default=None, help="Path to Showdown battle JSONL")
    parser.addoption("--num-battles", type=int, default=100, help="Number of battles to run")


@pytest.fixture(scope="session")
def tables4():
    """Gen 4 tables (built from hard-coded data, no Showdown install needed)."""
    return load_tables(4)


@pytest.fixture(scope="session")
def cfg4():
    return GenConfig.for_gen(4)
