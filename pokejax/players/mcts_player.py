"""
poke-env Player that uses GPU-parallel expectimax search for move selection.

Instead of picking moves from a single neural-network forward pass, this
player converts each game state into the pokejax engine, simulates all
possible (our_action × opponent_action) pairs in parallel on the GPU,
and picks the action with the highest expected value.

Usage:
    from pokejax.players.mcts_player import MctsPlayer
    player = MctsPlayer(
        checkpoint_path="checkpoints/ppo_latest.pkl",
        n_samples=16,
    )
    await player.battle_against(opponent, n_battles=10)
"""

from __future__ import annotations

import pickle
import time

import numpy as np
import jax
import jax.numpy as jnp

from poke_env.player import Player
from poke_env.environment import AbstractBattle

from pokejax.rl.model import PokeTransformer, create_model
from pokejax.data.tables import load_tables
from pokejax.env.pokejax_env import PokeJAXEnv, EnvState
from pokejax.search.battle_bridge import BattleBridge
from pokejax.search.expectimax import ExpectiMaxSearch, SearchResult
from pokejax.players.showdown_player import ObsBridge, N_TOKENS, FLOAT_DIM, N_ACTIONS, INT_IDS_PER_TOKEN


class MctsPlayer(Player):
    """Pokemon Showdown player using GPU-parallel expectimax search.

    For each turn:
      1. Convert poke-env Battle → pokejax BattleState (via BattleBridge)
      2. Run pre-compiled ExpectiMaxSearch on GPU
      3. Map best action back to poke-env BattleOrder

    Falls back to neural network policy if search fails.

    Parameters
    ----------
    checkpoint_path : str
        Path to model checkpoint (pickle with "params" key).
    gen : int
        Pokemon generation (default 4).
    n_samples : int
        RNG samples per action pair (default 16).
    opp_temperature : float
        Temperature for opponent policy modeling (default 0.5).
    verbose : bool
        Print search details per turn.
    **kwargs
        Passed to poke-env Player.
    """

    def __init__(
        self,
        checkpoint_path: str,
        gen: int = 4,
        n_samples: int = 16,
        opp_temperature: float = 0.5,
        verbose: bool = False,
        prebuilt_searcher: ExpectiMaxSearch | None = None,
        prebuilt_params=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.verbose = verbose
        self.gen = gen

        t0 = time.time()
        print("[MctsPlayer] Initializing...", flush=True)

        # Load tables
        print("[MctsPlayer] Loading tables...", flush=True)
        self.tables = load_tables(gen)

        # Load model + params
        print(f"[MctsPlayer] Loading checkpoint: {checkpoint_path}", flush=True)
        if prebuilt_params is not None:
            self.params = prebuilt_params
            arch = "transformer"
        else:
            with open(checkpoint_path, "rb") as f:
                ckpt = pickle.load(f)
            self.params = ckpt["params"]
            arch = ckpt.get("arch", "transformer")
        self.model = create_model(arch)

        # Initialize the pokejax environment (for engine simulation)
        print("[MctsPlayer] Creating pokejax environment...", flush=True)
        self.env = PokeJAXEnv(gen=gen)

        # Bridges: ObsBridge for obs + action mapping, BattleBridge for state conversion
        print("[MctsPlayer] Setting up bridges...", flush=True)
        self.obs_bridge = ObsBridge(self.tables)
        self.battle_bridge = BattleBridge(self.obs_bridge)

        # Search engine — use prebuilt if provided (already compiled)
        if prebuilt_searcher is not None:
            print("[MctsPlayer] Using prebuilt search engine.", flush=True)
            self.searcher = prebuilt_searcher
            self._warmup_done = True
        else:
            print("[MctsPlayer] Building search kernel...", flush=True)
            self.searcher = ExpectiMaxSearch(
                env=self.env,
                model=self.model,
                params=self.params,
                n_samples=n_samples,
                opp_temperature=opp_temperature,
                warmup=False,
            )
            self._warmup_done = False

        # RNG state
        self._rng_key = jax.random.PRNGKey(42)

        # JIT-compiled fallback forward pass
        print("[MctsPlayer] Compiling fallback forward pass...", flush=True)

        @jax.jit
        def _forward(params, int_ids, float_feats, legal_mask):
            log_probs, _, value = self.model.apply(
                params, int_ids[None], float_feats[None], legal_mask[None],
            )
            return log_probs[0], value[0]

        self._forward = _forward

        # Warm up fallback
        dummy_int = jnp.zeros((N_TOKENS, INT_IDS_PER_TOKEN), dtype=jnp.int32)
        dummy_float = jnp.zeros((N_TOKENS, FLOAT_DIM), dtype=jnp.float32)
        dummy_mask = jnp.ones(N_ACTIONS, dtype=jnp.float32)
        _ = self._forward(self.params, dummy_int, dummy_float, dummy_mask)

        print(f"[MctsPlayer] Ready! Init took {time.time() - t0:.1f}s", flush=True)

    def warmup(self):
        """Force JIT compilation of search kernels. Call before ladder()."""
        if self._warmup_done:
            return
        print("[MctsPlayer] Warming up search kernels...", flush=True)
        t0 = time.time()
        dummy_key = jax.random.PRNGKey(0)
        dummy_state, _ = self.env.reset(dummy_key)
        dummy_keys = jax.random.split(dummy_key, self.searcher.n_total)
        # Kernel 1
        print("[MctsPlayer]   1/3 root_eval...", flush=True)
        out = self.searcher._root_eval(self.params, dummy_state)
        out[0].block_until_ready()
        # Kernel 2
        print("[MctsPlayer]   2/3 simulate...", flush=True)
        lv = self.searcher._simulate(self.params, dummy_state, dummy_keys)
        lv.block_until_ready()
        # Kernel 3
        print("[MctsPlayer]   3/3 aggregate...", flush=True)
        av = self.searcher._aggregate(lv, out[3], out[4], out[2])
        av.block_until_ready()
        self._warmup_done = True
        print(f"[MctsPlayer] Warmup done! Took {time.time() - t0:.1f}s", flush=True)

    def choose_move(self, battle: AbstractBattle):
        """Choose a move using expectimax search."""
        if not self._warmup_done:
            self.warmup()
        try:
            return self._choose_move_search(battle)
        except Exception as e:
            if self.verbose:
                import traceback
                traceback.print_exc()
                print(f"  Search failed: {e}, falling back to policy")
            try:
                return self._choose_move_policy(battle)
            except Exception:
                return self.choose_default_move()

    def _choose_move_search(self, battle: AbstractBattle):
        """Use expectimax search to select the best action."""
        t0 = time.time()

        # Step 1: Convert battle → pokejax state
        self._rng_key, bridge_key, search_key = jax.random.split(self._rng_key, 3)
        battle_state, reveal_state = self.battle_bridge.battle_to_state(
            battle, rng_key=bridge_key,
        )
        env_state = EnvState(battle=battle_state, reveal=reveal_state)

        # Step 2: Run pre-compiled search
        result: SearchResult = self.searcher.search(env_state, search_key)
        action = result.best_action

        t1 = time.time()

        if self.verbose:
            self._print_search_info(battle, result, t1 - t0)

        # Step 3: Map action → poke-env order
        return self._action_to_order(battle, action)

    def _choose_move_policy(self, battle: AbstractBattle):
        """Fallback: neural network policy (no search)."""
        obs = self.obs_bridge.build_obs(battle)

        int_ids = jnp.array(obs["int_ids"])
        float_feats = jnp.array(obs["float_feats"])
        legal_mask = jnp.array(obs["legal_mask"])

        trapped = battle.trapped if hasattr(battle, 'trapped') else False
        if trapped:
            legal_mask = legal_mask.at[4:].set(0.0)
            if legal_mask[:4].sum() == 0:
                legal_mask = legal_mask.at[0].set(1.0)

        log_probs, _ = self._forward(self.params, int_ids, float_feats, legal_mask)
        probs = np.exp(np.array(log_probs))
        masked_probs = probs * np.array(obs["legal_mask"])
        action = int(np.argmax(masked_probs))

        return self._action_to_order(battle, action)

    def _action_to_order(self, battle: AbstractBattle, action: int):
        """Map a pokejax action (0-9) to a poke-env BattleOrder."""
        available_moves = battle.available_moves
        available_switches = battle.available_switches
        active_pokemon = battle.active_pokemon

        # Ensure obs_bridge has the latest team/move mapping
        self.obs_bridge.build_obs(battle)
        own_team = self.obs_bridge._last_own_team
        own_move_list = self.obs_bridge._last_own_move_list

        if action < 4:
            if action < len(own_move_list) and own_move_list[action] is not None:
                chosen_id = own_move_list[action].id
                for m in available_moves:
                    if m.id == chosen_id:
                        return self.create_order(m)
            if available_moves:
                return self.create_order(available_moves[0])
        else:
            slot = action - 4
            if slot < len(own_team) and own_team[slot] is not None:
                target = own_team[slot]
                # Safety: never switch to the active pokemon
                if target is active_pokemon:
                    print(f"  [BUG] Expectimax chose switch to active "
                          f"{active_pokemon.species} (slot {slot}), falling back")
                elif target in available_switches:
                    return self.create_order(target)
            if available_switches:
                return self.create_order(available_switches[0])

        if available_moves:
            return self.create_order(available_moves[0])
        if available_switches:
            return self.create_order(available_switches[0])
        return self.choose_default_move()

    def _print_search_info(self, battle: AbstractBattle, result: SearchResult, dt: float):
        """Print search diagnostics."""
        action_names = [f"move{i}" if i < 4 else f"switch{i-4}" for i in range(N_ACTIONS)]

        print(f"  Turn {battle.turn}: search={dt*1000:.0f}ms "
              f"| root_V={result.root_value:.3f}")

        vals = result.action_values.copy()
        legal = vals > -1e8
        for idx in np.argsort(-vals):
            if not legal[idx]:
                continue
            marker = " <--" if idx == result.best_action else ""
            print(f"    {action_names[idx]:>8s}: V={vals[idx]:+.4f}{marker}")
