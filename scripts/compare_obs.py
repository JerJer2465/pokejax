#!/usr/bin/env python3
"""
Compare engine obs vs bridge obs side-by-side.
Runs Showdown games and on each choose_move, builds obs from both
the bridge and compares feature values to find systematic differences.
"""
import asyncio
import sys
import numpy as np

sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1, closefd=False)


async def main():
    from poke_env import LocalhostServerConfiguration, AccountConfiguration
    from poke_env.player import RandomPlayer
    from poke_env.environment import AbstractBattle
    from pokejax.players.showdown_player import PokejaxPlayer, ObsBridge
    import jax.numpy as jnp

    # Feature offset names for readable output
    FEATURE_NAMES = {}
    # From obs_builder.py
    FEATURE_NAMES[0] = "hp_frac"
    for i in range(10):
        FEATURE_NAMES[1 + i] = f"hp_bin_{i}"
    for i, s in enumerate(['hp', 'atk', 'def', 'spa', 'spd', 'spe']):
        FEATURE_NAMES[11 + i] = f"base_{s}"
    for i in range(7):
        for j in range(13):
            FEATURE_NAMES[17 + i*13 + j] = f"boost_{['atk','def','spa','spd','spe','acc','eva'][i]}_{j-6}"
    for i, s in enumerate(['none', 'brn', 'psn', 'tox', 'slp', 'frz', 'par']):
        FEATURE_NAMES[108 + i] = f"status_{s}"
    for i in range(27):
        FEATURE_NAMES[115 + i] = f"volatile_{i}"
    for i in range(18):
        FEATURE_NAMES[142 + i] = f"type1_{i}"
    for i in range(18):
        FEATURE_NAMES[160 + i] = f"type2_{i}"
    FEATURE_NAMES[178] = "is_fainted"
    FEATURE_NAMES[179] = "is_active"
    for i in range(6):
        FEATURE_NAMES[180 + i] = f"slot_{i}"
    FEATURE_NAMES[186] = "is_own"
    for m in range(4):
        base = 187 + m * 45
        for i in range(8):
            FEATURE_NAMES[base + i] = f"move{m}_bp_bin_{i}"
        for i in range(6):
            FEATURE_NAMES[base + 8 + i] = f"move{m}_acc_bin_{i}"
        for i in range(18):
            FEATURE_NAMES[base + 14 + i] = f"move{m}_type_{i}"
        for i in range(3):
            FEATURE_NAMES[base + 32 + i] = f"move{m}_cat_{i}"
        for i in range(8):
            FEATURE_NAMES[base + 35 + i] = f"move{m}_pri_{i}"
        FEATURE_NAMES[base + 43] = f"move{m}_pp"
        FEATURE_NAMES[base + 44] = f"move{m}_known"
    FEATURE_NAMES[367] = "sleep_bin_0"
    FEATURE_NAMES[368] = "sleep_bin_1"
    FEATURE_NAMES[369] = "sleep_bin_2"
    FEATURE_NAMES[370] = "sleep_bin_3"
    FEATURE_NAMES[371] = "rest_bin_0"
    FEATURE_NAMES[372] = "rest_bin_1"
    FEATURE_NAMES[373] = "rest_bin_2"
    FEATURE_NAMES[374] = "sub_frac"
    FEATURE_NAMES[375] = "force_trap"
    for i in range(4):
        FEATURE_NAMES[376 + i] = f"mov_dis_{i}"
    FEATURE_NAMES[380] = "conf_bin_0"
    FEATURE_NAMES[381] = "conf_bin_1"
    FEATURE_NAMES[382] = "conf_bin_2"
    FEATURE_NAMES[383] = "conf_bin_3"
    FEATURE_NAMES[384] = "taunt"
    FEATURE_NAMES[385] = "encore"
    FEATURE_NAMES[386] = "yawn"
    FEATURE_NAMES[387] = "level"
    FEATURE_NAMES[388] = "perish_bin_0"
    FEATURE_NAMES[389] = "perish_bin_1"
    FEATURE_NAMES[390] = "perish_bin_2"
    FEATURE_NAMES[391] = "perish_bin_3"
    FEATURE_NAMES[392] = "protect"
    FEATURE_NAMES[393] = "locked_mov"

    # Collect diffs
    all_diffs = {}  # feature_idx -> list of (bridge_val, turn, token, species)
    turn_count = [0]

    original_choose = PokejaxPlayer._choose_move_impl

    def compare_choose_move(self, battle: AbstractBattle):
        turn_count[0] += 1
        obs = self.obs_bridge.build_obs(battle)

        # Check the active own Pokemon token specifically
        own_active = battle.active_pokemon
        species = own_active.species if own_active else "?"

        # Look at tokens 1-6 (own team)
        for tok in range(1, 7):
            ff = obs["float_feats"][tok]
            ii = obs["int_ids"][tok]

            # Check if this is the active token
            is_active = ff[179] > 0.5
            is_own = ff[186] > 0.5

            if not is_own:
                continue

            tok_species_id = int(ii[0])
            if tok_species_id == 0:
                continue

            # Check specific features for issues
            hp_frac = ff[0]
            level = ff[387]

            # Key checks:
            # 1. Is sleep_bin correct? (should have exactly one 1.0)
            sleep_bins = ff[367:371]
            sleep_sum = sleep_bins.sum()

            # 2. Is rest_bin correct?
            rest_bins = ff[371:374]
            rest_sum = rest_bins.sum()

            # 3. Is status one-hot?
            status = ff[108:115]
            status_sum = status.sum()

            # 4. Is confusion bin one-hot?
            conf_bins = ff[380:384]
            conf_sum = conf_bins.sum()

            # 5. Is perish bin one-hot?
            perish_bins = ff[388:392]
            perish_sum = perish_bins.sum()

            # 6. Are move known flags correct?
            for m in range(4):
                known = ff[187 + m*45 + 44]
                mid = int(ii[m+1])

            # Report issues
            issues = []
            if abs(sleep_sum - 1.0) > 0.01:
                issues.append(f"sleep_bin sum={sleep_sum:.2f} (should be 1)")
            if abs(rest_sum - 1.0) > 0.01:
                issues.append(f"rest_bin sum={rest_sum:.2f} (should be 1)")
            if abs(status_sum - 1.0) > 0.01:
                issues.append(f"status sum={status_sum:.2f} (should be 1)")
            if abs(conf_sum - 1.0) > 0.01:
                issues.append(f"conf_bin sum={conf_sum:.2f} (should be 1)")
            if abs(perish_sum - 1.0) > 0.01:
                issues.append(f"perish_bin sum={perish_sum:.2f} (should be 1)")
            if level < 0.01 and hp_frac > 0:
                issues.append(f"level=0 but alive")

            if issues and is_active:
                print(f"  Turn {battle.turn} token {tok} ({species}): {'; '.join(issues)}")

        # Also check field token
        ff0 = obs["float_feats"][0]
        # Fainted counts
        own_fainted_frac = ff0[58]
        opp_fainted_frac = ff0[59]

        # Toxic count bins
        toxic_own = ff0[60:65]
        toxic_opp = ff0[65:70]
        if toxic_own.sum() < 0.01:
            pass  # OK if no toxic bin set — but engine always sets bin 0 if no toxic
        if toxic_opp.sum() < 0.01:
            pass  # same

        return original_choose(self, battle)

    PokejaxPlayer._choose_move_impl = compare_choose_move

    # Also check the value directly
    original_choose2 = PokejaxPlayer.choose_move
    value_list = []

    def logging_choose(self, battle):
        try:
            obs = self.obs_bridge.build_obs(battle)
            int_ids = jnp.array(obs["int_ids"])
            float_feats = jnp.array(obs["float_feats"])
            legal_mask = jnp.array(obs["legal_mask"])
            _, value = self._forward(self.params, int_ids, float_feats, legal_mask)
            value_list.append(float(value))
        except:
            pass
        return original_choose2(self, battle)

    PokejaxPlayer.choose_move = logging_choose

    player = PokejaxPlayer(
        checkpoint_path="checkpoints/bc_final.pkl",
        gen=4, temperature=0.0, verbose=False,
        account_configuration=AccountConfiguration("CompareBot", None),
        server_configuration=LocalhostServerConfiguration,
        battle_format="gen4randombattle",
    )
    opponent = RandomPlayer(
        account_configuration=AccountConfiguration("RandBotC", None),
        server_configuration=LocalhostServerConfiguration,
        battle_format="gen4randombattle",
    )

    print("Playing 5 games, checking obs encoding...")
    await player.battle_against(opponent, n_battles=5)

    wins = player.n_won_battles
    losses = player.n_lost_battles
    total = wins + losses
    print(f"\nResults: {wins}W / {losses}L ({wins}/{total} = {wins/max(total,1):.0%})")
    if value_list:
        print(f"Value stats: mean={np.mean(value_list):.3f} "
              f"min={np.min(value_list):.3f} max={np.max(value_list):.3f}")
        print(f"  Negative values: {sum(1 for v in value_list if v < 0)}/{len(value_list)}")

    # Now check what the ENGINE obs looks like for a fresh game
    print("\n--- Engine obs check ---")
    from pokejax.rl.obs_builder import build_obs as engine_build_obs
    from pokejax.env.pokejax_env import PokejaxEnv
    from pokejax.data.tables import load_tables
    import jax

    tables = load_tables(4)
    env = PokejaxEnv(tables)
    rng = jax.random.PRNGKey(42)
    state, reveal = env.reset(rng)

    # Build engine obs for player 0
    engine_obs = engine_build_obs(state, reveal, player=0, tables=tables)
    eng_ff = np.array(engine_obs["float_feats"])
    eng_ii = np.array(engine_obs["int_ids"])

    print(f"Engine obs shapes: ff={eng_ff.shape} ii={eng_ii.shape}")

    # Check own active token (token 1, slot 0)
    tok1 = eng_ff[1]
    print(f"\nEngine token 1 (own active, slot 0):")
    print(f"  hp_frac={tok1[0]:.3f}")
    print(f"  hp_bin={tok1[1:11].tolist()}")
    print(f"  base_stats={tok1[11:17].tolist()}")
    print(f"  status={tok1[108:115].tolist()}")
    print(f"  type1={np.argmax(tok1[142:160])} type2={'none' if tok1[160:178].sum()<0.5 else np.argmax(tok1[160:178])}")
    print(f"  is_fainted={tok1[178]:.0f} is_active={tok1[179]:.0f}")
    print(f"  slot={tok1[180:186].tolist()}")
    print(f"  is_own={tok1[186]:.0f}")
    print(f"  level={tok1[387]:.3f}")
    print(f"  sleep_bin={tok1[367:371].tolist()}")
    print(f"  rest_bin={tok1[371:374].tolist()}")
    print(f"  conf_bin={tok1[380:384].tolist()}")
    print(f"  perish_bin={tok1[388:392].tolist()}")
    print(f"  protect={tok1[392]:.3f}")
    print(f"  locked_mov={tok1[393]:.3f}")
    print(f"  move0_known={tok1[187+44]:.0f} move1_known={tok1[187+45+44]:.0f}")
    print(f"  move0_pp={tok1[187+43]:.3f} move1_pp={tok1[187+45+43]:.3f}")

    # Check field token
    ftok = eng_ff[0]
    print(f"\nEngine field token:")
    print(f"  weather={ftok[0:5].tolist()}")
    print(f"  weather_turns={ftok[5:13].tolist()}")
    print(f"  fainted_frac own={ftok[58]:.3f} opp={ftok[59]:.3f}")
    print(f"  toxic_own={ftok[60:65].tolist()}")
    print(f"  toxic_opp={ftok[65:70].tolist()}")
    print(f"  tailwind={ftok[70:72].tolist()}")
    print(f"  wish={ftok[72:74].tolist()}")
    print(f"  safeguard={ftok[74:76].tolist()}")
    print(f"  mist={ftok[76:78].tolist()}")
    print(f"  lucky_chant={ftok[78:80].tolist()}")
    print(f"  gravity_turns={ftok[80:84].tolist()}")


if __name__ == "__main__":
    asyncio.run(main())
