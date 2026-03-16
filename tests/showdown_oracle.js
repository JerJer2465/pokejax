/**
 * Pokemon Showdown Oracle — headless reference engine for parity tests.
 *
 * Reads JSON scenario descriptors from stdin (one per line) and writes
 * JSON results to stdout.  Each scenario specifies teams, field state,
 * and a sequence of actions; the oracle runs them through PS's battle
 * engine and returns the resulting state.
 *
 * Usage:
 *   node showdown_oracle.js < scenarios.jsonl > results.jsonl
 *
 * Or interactively via subprocess from Python (test_showdown_parity.py).
 *
 * Requirements:
 *   npm install pokemon-showdown   (or clone + npm install)
 *   Set POKEMON_SHOWDOWN_PATH env var to the repo root.
 */

'use strict';

const path = require('path');
const readline = require('readline');

// Locate Pokemon Showdown
const PS_PATH = process.env.POKEMON_SHOWDOWN_PATH
    || path.join(__dirname, '..', '..', 'pokemon-showdown');

let Sim;
try {
    // Try as installed package first
    Sim = require(path.join(PS_PATH, 'sim'));
} catch {
    try {
        Sim = require('pokemon-showdown/sim');
    } catch {
        console.error(`Cannot find Pokemon Showdown at ${PS_PATH}`);
        console.error('Set POKEMON_SHOWDOWN_PATH or npm install pokemon-showdown');
        process.exit(1);
    }
}

const { Battle, Dex } = Sim;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makePokemonSet(spec) {
    /**
     * spec: { species, moves, ability?, item?, level?, evs?, ivs?, nature? }
     */
    return {
        species: spec.species || 'Bulbasaur',
        moves: spec.moves || ['Tackle'],
        ability: spec.ability || '',
        item: spec.item || '',
        level: spec.level || 100,
        evs: spec.evs || { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 },
        ivs: spec.ivs || { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 },
        nature: spec.nature || 'Hardy',
        gender: spec.gender || '',
    };
}

function extractPokemonState(pokemon) {
    return {
        species: pokemon.species.name || pokemon.species,
        hp: pokemon.hp,
        maxhp: pokemon.maxhp,
        status: pokemon.status || '',
        statusData: pokemon.statusData ? {
            toxicTurns: pokemon.statusData.toxicTurns || 0,
            sleepTurns: pokemon.statusData.sleepTurns || 0,
        } : {},
        boosts: { ...pokemon.boosts },
        item: pokemon.item || '',
        ability: pokemon.ability || '',
        volatiles: Object.keys(pokemon.volatiles || {}),
        moves: pokemon.moveSlots ? pokemon.moveSlots.map(m => ({
            id: m.id,
            pp: m.pp,
            maxpp: m.maxpp,
        })) : [],
        fainted: pokemon.fainted || false,
        types: pokemon.types ? [...pokemon.types] : [],
    };
}

function extractSideState(side) {
    return {
        name: side.name,
        pokemon: side.pokemon.map(extractPokemonState),
        active: side.active.map(p => p ? extractPokemonState(p) : null),
        sideConditions: Object.fromEntries(
            Object.entries(side.sideConditions || {}).map(
                ([k, v]) => [k, { layers: v.layers || 0, duration: v.duration || 0 }]
            )
        ),
        pokemonLeft: side.pokemonLeft,
    };
}

function extractFieldState(field) {
    return {
        weather: field.weather || '',
        weatherDuration: field.weatherState ? field.weatherState.duration : 0,
        terrain: field.terrain || '',
        terrainDuration: field.terrainState ? field.terrainState.duration : 0,
        pseudoWeather: Object.fromEntries(
            Object.entries(field.pseudoWeather || {}).map(
                ([k, v]) => [k, { duration: v.duration || 0 }]
            )
        ),
    };
}

function extractBattleState(battle) {
    return {
        turn: battle.turn,
        ended: battle.ended,
        winner: battle.winner || '',
        p1: extractSideState(battle.sides[0]),
        p2: extractSideState(battle.sides[1]),
        field: extractFieldState(battle.field),
    };
}

// ---------------------------------------------------------------------------
// Scenario runner
// ---------------------------------------------------------------------------

function runScenario(scenario) {
    /**
     * scenario: {
     *   format: "gen4randombattle" | "gen4ou" | etc,
     *   p1team: [PokemonSpec, ...],
     *   p2team: [PokemonSpec, ...],
     *   seed: [s1, s2, s3, s4],   // optional PRNG seed
     *   actions: [
     *     { type: "move", p1: "move 1", p2: "move 1" },
     *     { type: "switch", p1: "switch 2", p2: "move 1" },
     *     ...
     *   ],
     *   // Optional: inject state modifications before actions
     *   inject?: {
     *     p1_status?: { slot: 0, status: "brn" },
     *     p2_status?: { slot: 0, status: "par" },
     *     weather?: "SunnyDay",
     *     p1_side_conditions?: { stealthrock: { layers: 1 } },
     *     ...
     *   },
     *   // What to extract
     *   query: "full_state" | "damage_calc" | "turn_order",
     * }
     */
    const format = scenario.format || 'gen4ou';

    const battle = new Battle({
        formatid: format,
        // Disable RNG randomness where we can control seeds
        seed: scenario.seed || [1, 2, 3, 4],
        debug: false,
    });

    // Set teams
    const p1team = (scenario.p1team || []).map(makePokemonSet);
    const p2team = (scenario.p2team || []).map(makePokemonSet);

    battle.setPlayer('p1', { team: Dex.packTeam(p1team) });
    battle.setPlayer('p2', { team: Dex.packTeam(p2team) });

    // Capture initial state
    const states = [extractBattleState(battle)];

    // Inject state modifications if requested
    if (scenario.inject) {
        const inj = scenario.inject;

        if (inj.p1_status) {
            const pokemon = battle.sides[0].pokemon[inj.p1_status.slot];
            if (inj.p1_status.status) {
                pokemon.setStatus(inj.p1_status.status);
            }
        }
        if (inj.p2_status) {
            const pokemon = battle.sides[1].pokemon[inj.p2_status.slot];
            if (inj.p2_status.status) {
                pokemon.setStatus(inj.p2_status.status);
            }
        }
        if (inj.weather) {
            battle.field.setWeather(inj.weather);
        }
        if (inj.p1_boosts) {
            const active = battle.sides[0].active[0];
            if (active) active.setBoost(inj.p1_boosts);
        }
        if (inj.p2_boosts) {
            const active = battle.sides[1].active[0];
            if (active) active.setBoost(inj.p2_boosts);
        }

        states.push(extractBattleState(battle));
    }

    // Execute actions
    for (const action of (scenario.actions || [])) {
        if (action.p1 && action.p2) {
            battle.makeChoices(action.p1, action.p2);
        } else if (action.p1) {
            battle.makeChoices(action.p1, 'default');
        } else if (action.p2) {
            battle.makeChoices('default', action.p2);
        }
        states.push(extractBattleState(battle));
    }

    // Extract log for debugging
    const log = battle.log ? battle.log.join('\n') : '';

    return {
        scenario_id: scenario.id || 'unknown',
        states: states,
        log: log,
        inputLog: battle.inputLog ? battle.inputLog.join('\n') : '',
    };
}

// ---------------------------------------------------------------------------
// Main: read scenarios from stdin, write results to stdout
// ---------------------------------------------------------------------------

const rl = readline.createInterface({ input: process.stdin });

rl.on('line', (line) => {
    try {
        const scenario = JSON.parse(line);
        const result = runScenario(scenario);
        console.log(JSON.stringify(result));
    } catch (err) {
        console.log(JSON.stringify({
            error: err.message,
            stack: err.stack,
        }));
    }
});

rl.on('close', () => {
    process.exit(0);
});
