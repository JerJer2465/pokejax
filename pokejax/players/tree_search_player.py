"""
poke-env Player that uses GPU-accelerated MCTS for move selection.

Unlike the depth-1 expectimax in mcts_player.py, this uses true Monte Carlo
Tree Search with PUCT (AlphaZero-style) for multi-turn lookahead. The neural
network provides policy priors and value estimates to guide the tree search.

Designed to run in WSL with CUDA for maximum GPU throughput.

Usage:
    from pokejax.players.tree_search_player import TreeSearchPlayer
    player = TreeSearchPlayer(
        checkpoint_path="checkpoints/ppo_best.pkl",
        n_simulations=128,
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

from pokejax.rl.model import PokeTransformer
from pokejax.data.tables import load_tables
from pokejax.env.pokejax_env import PokeJAXEnv, EnvState
from pokejax.search.battle_bridge import BattleBridge
from pokejax.search.mcts import MCTSSearch, SearchResult
from pokejax.players.showdown_player import (
    ObsBridge, N_TOKENS, FLOAT_DIM, N_ACTIONS, INT_IDS_PER_TOKEN,
)


class TreeSearchPlayer(Player):
    """Pokemon Showdown player using GPU-accelerated MCTS tree search.

    For each turn:
      1. Convert poke-env Battle -> pokejax BattleState (via BattleBridge)
      2. Run MCTS search with PUCT on GPU (multi-turn lookahead)
      3. Map best action back to poke-env BattleOrder

    Falls back to neural network policy if search fails.

    Parameters
    ----------
    checkpoint_path : str
        Path to model checkpoint (pickle with "params" key).
    gen : int
        Pokemon generation (default 4).
    n_simulations : int
        MCTS simulations per move (default 128).
    c_puct : float
        PUCT exploration constant (default 2.5).
    opp_temperature : float
        Temperature for opponent policy sampling (default 0.5).
    max_depth : int
        Maximum search depth in turns (default 10).
    batch_size : int
        Batch size for GPU leaf expansion (default 8).
    use_batched : bool
        Use virtual-loss batched search for path diversity (default False).
    verbose : bool
        Print search details per turn.
    prebuilt_searcher : MCTSSearch or None
        Pre-compiled searcher (avoids re-compilation).
    prebuilt_params : dict or None
        Pre-loaded model params.
    **kwargs
        Passed to poke-env Player.
    """

    def __init__(
        self,
        checkpoint_path: str,
        gen: int = 4,
        n_simulations: int = 128,
        c_puct: float = 2.5,
        opp_temperature: float = 0.5,
        max_depth: int = 10,
        dirichlet_alpha: float = 0.3,
        dirichlet_frac: float = 0.25,
        batch_size: int = 8,
        use_batched: bool = False,
        verbose: bool = False,
        prebuilt_searcher: MCTSSearch | None = None,
        prebuilt_params=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.verbose = verbose
        self.gen = gen
        self.use_batched = use_batched
        self.n_simulations = n_simulations

        t0 = time.time()
        print("[TreeSearch] Initializing...", flush=True)

        # Load tables
        self.tables = load_tables(gen)

        # Load model + params
        print(f"[TreeSearch] Loading checkpoint: {checkpoint_path}", flush=True)
        self.model = PokeTransformer()
        if prebuilt_params is not None:
            self.params = prebuilt_params
        else:
            with open(checkpoint_path, "rb") as f:
                ckpt = pickle.load(f)
            self.params = ckpt["params"]

        # pokejax environment (for engine simulation)
        self.env = PokeJAXEnv(gen=gen)

        # Bridges
        self.obs_bridge = ObsBridge(self.tables)
        self.battle_bridge = BattleBridge(self.obs_bridge)

        # MCTS Search engine
        if prebuilt_searcher is not None:
            print("[TreeSearch] Using prebuilt MCTS engine.", flush=True)
            self.searcher = prebuilt_searcher
        else:
            print("[TreeSearch] Building MCTS engine...", flush=True)
            self.searcher = MCTSSearch(
                env=self.env,
                model=self.model,
                params=self.params,
                n_simulations=n_simulations,
                c_puct=c_puct,
                opp_temperature=opp_temperature,
                max_depth=max_depth,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_frac=dirichlet_frac,
                batch_size=batch_size,
                warmup=True,
            )

        # RNG state
        self._rng_key = jax.random.PRNGKey(42)
        self._search_count = 0

        # Fallback forward pass (for when search fails)
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

        print(f"[TreeSearch] Ready! Init took {time.time() - t0:.1f}s", flush=True)

    def _stale_request_type(self, battle: AbstractBattle) -> int:
        """Detect stale request data after switches.

        Returns 0 (not stale), 1 (move mismatch), 2 (forced switch stale).
        See PokejaxPlayer._stale_request_type for full documentation.
        """
        active = battle.active_pokemon
        if not active:
            return 0
        available_moves = battle.available_moves
        available_switches = battle.available_switches
        if available_moves and active.moves:
            active_move_ids = set(m.id for m in active.moves.values())
            active_hp = set(active_move_ids)
            for mid in active_move_ids:
                if mid.startswith('hiddenpower'):
                    active_hp.add('hiddenpower')
            for m in available_moves:
                mid = m.id
                norm = mid if not mid.startswith('hiddenpower') else 'hiddenpower'
                if mid not in active_hp and norm not in active_hp:
                    return 1
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

        Sends a deliberately wrong command so PS rejects it and sends
        the correct request, avoiding running MCTS on stale data.
        """
        if (not from_teampreview_request and not maybe_default_order
                and not battle.teampreview):
            stale_type = self._stale_request_type(battle)
            if stale_type == 1:
                if self.verbose:
                    print(f"  [STALE] turn={battle.turn}: sending rejectable move",
                          flush=True)
                available_moves = battle.available_moves
                if available_moves:
                    message = self.create_order(available_moves[0]).message
                    await self.ps_client.send_message(
                        message, battle.battle_tag)
                    return
            elif stale_type == 2:
                if self.verbose:
                    print(f"  [STALE] turn={battle.turn}: sending rejectable switch",
                          flush=True)
                message = self.create_order(battle.active_pokemon).message
                await self.ps_client.send_message(message, battle.battle_tag)
                return
        await super()._handle_battle_request(
            battle,
            from_teampreview_request=from_teampreview_request,
            maybe_default_order=maybe_default_order,
        )

    def choose_move(self, battle: AbstractBattle):
        """Choose a move using MCTS search."""
        try:
            return self._choose_move_search(battle)
        except Exception as e:
            if self.verbose:
                import traceback
                traceback.print_exc()
                print(f"  MCTS failed: {e}, falling back to policy", flush=True)
            try:
                return self._choose_move_policy(battle)
            except Exception:
                return self.choose_default_move()

    def _choose_move_search(self, battle: AbstractBattle):
        """Use MCTS to select the best action."""
        t0 = time.time()

        # Build obs to populate _last_own_team / _last_own_move_list
        obs = self.obs_bridge.build_obs(battle)

        # Handle trapped status
        trapped = battle.trapped if hasattr(battle, 'trapped') else False
        if trapped:
            obs["legal_mask"][4:] = 0.0
            if obs["legal_mask"][:4].sum() == 0:
                available_move_names = set(m.id for m in battle.available_moves)
                for i, m in enumerate(self.obs_bridge._last_own_move_list[:4]):
                    if m is not None and m.id in available_move_names:
                        obs["legal_mask"][i] = 1.0
            if obs["legal_mask"].sum() == 0:
                obs["legal_mask"][0] = 1.0

        # Skip search if only one legal action
        legal_actions = np.where(obs["legal_mask"] > 0)[0]
        if len(legal_actions) <= 1:
            action = legal_actions[0] if len(legal_actions) == 1 else 0
            if self.verbose:
                print(f"  Turn {battle.turn}: only 1 legal action -> {action}",
                      flush=True)
            return self._action_to_order(battle, action)

        # Convert battle -> pokejax state
        self._rng_key, bridge_key, search_key = jax.random.split(self._rng_key, 3)
        battle_state, reveal_state = self.battle_bridge.battle_to_state(
            battle, rng_key=bridge_key,
        )
        env_state = EnvState(battle=battle_state, reveal=reveal_state)

        # Run MCTS
        if self.use_batched:
            result: SearchResult = self.searcher.search_batched(env_state, search_key)
        else:
            result: SearchResult = self.searcher.search(env_state, search_key)

        action = result.best_action
        self._search_count += 1

        # Safety: override if MCTS picked an illegal action from poke-env's view
        if obs["legal_mask"][action] <= 0:
            visits = result.search_policy.copy()
            visits[obs["legal_mask"] <= 0] = -1
            action = int(np.argmax(visits))
            if self.verbose:
                print(f"  MCTS action overridden to {action} (legality)", flush=True)

        dt = time.time() - t0
        if self.verbose:
            self._print_search_info(battle, result, dt, trapped)

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

        # Ensure obs_bridge has the latest mapping
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
                    if self.verbose:
                        print(f"  [BUG] MCTS chose switch to active "
                              f"{active_pokemon.species} (slot {slot}), "
                              f"falling back")
                elif target in available_switches:
                    return self.create_order(target)
            if available_switches:
                return self.create_order(available_switches[0])

        if available_moves:
            return self.create_order(available_moves[0])
        if available_switches:
            return self.create_order(available_switches[0])
        return self.choose_default_move()

    def _print_search_info(
        self, battle: AbstractBattle, result: SearchResult,
        dt: float, trapped: bool = False,
    ):
        """Print MCTS search diagnostics."""
        trap_str = " [TRAPPED]" if trapped else ""
        print(f"  Turn {battle.turn}: MCTS {self.n_simulations} sims "
              f"in {dt*1000:.0f}ms | root_V={result.root_value:.3f}{trap_str}",
              flush=True)

        vals = result.action_values.copy()
        policy = result.search_policy.copy()
        legal = vals > -1e8

        for idx in np.argsort(-policy):
            if not legal[idx]:
                continue
            name = f"move{idx}" if idx < 4 else f"switch{idx - 4}"
            marker = " <--" if idx == result.best_action else ""
            print(f"    {name:>8s}: visits={policy[idx]*100:5.1f}% "
                  f"Q={vals[idx]:+.4f}{marker}", flush=True)
