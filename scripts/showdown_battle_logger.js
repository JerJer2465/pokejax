/**
 * Showdown Battle Logger for Differential Testing
 *
 * Runs N Gen 4 random battles in Showdown's sim with deterministic seeds,
 * logging teams + per-turn actions + resulting state as JSON lines.
 *
 * Usage: node scripts/showdown_battle_logger.js [N_BATTLES] [OUTPUT_FILE]
 */

const path = require('path');

// Point to the Showdown dist
const PS_PATH = path.resolve(__dirname, '../../PokemonShowdownClaude/pokemon-showdown');
const sim = require(path.join(PS_PATH, 'dist/sim'));

const { BattleStream, getPlayerStreams, Teams } = sim;
const { Battle } = sim;
const { PRNG } = sim;
const { RandomPlayerAI } = require(path.join(PS_PATH, 'dist/sim/tools/random-player-ai'));

const N_BATTLES = parseInt(process.argv[2]) || 100;
const OUTPUT_FILE = process.argv[3] || path.join(__dirname, '..', 'data', 'showdown_battles.jsonl');

const fs = require('fs');

/**
 * Extract Pokemon state from a Showdown Pokemon object.
 */
function extractPokemonState(pokemon) {
    if (!pokemon) return null;
    return {
        species: pokemon.species?.id || pokemon.speciesData?.id || '',
        hp: pokemon.hp,
        maxhp: pokemon.maxhp,
        status: pokemon.status || '',
        statusData: pokemon.statusData?.turns || 0,
        boosts: { ...pokemon.boosts },
        ability: pokemon.ability || pokemon.baseAbility || '',
        item: pokemon.item || '',
        moves: pokemon.moveSlots?.map(m => ({
            id: m.id,
            pp: m.pp,
            maxpp: m.maxpp,
        })) || [],
        level: pokemon.level,
        types: [...(pokemon.types || [])],
        fainted: pokemon.fainted || false,
        isActive: pokemon.isActive || false,
        volatiles: Object.keys(pokemon.volatiles || {}),
    };
}

/**
 * Extract full battle state snapshot.
 */
function extractBattleState(battle) {
    const sides = battle.sides.map(side => {
        // Extract side conditions with layer counts / turn data
        const sc = {};
        for (const [key, val] of Object.entries(side.sideConditions || {})) {
            sc[key] = {
                layers: val.layers || (val['-1']?.layers) || 1,
                duration: val.duration || 0,
            };
        }
        return {
            name: side.name,
            pokemon: side.pokemon.map(p => extractPokemonState(p)),
            active: side.active?.map(p => p ? p.species?.id || '' : '') || [],
            pokemonLeft: side.pokemonLeft,
            sideConditions: sc,
        };
    });
    return {
        turn: battle.turn,
        weather: battle.field?.weatherState?.id || '',
        weatherTurns: battle.field?.weatherState?.duration || 0,
        terrain: battle.field?.terrainState?.id || '',
        terrainTurns: battle.field?.terrainState?.duration || 0,
        trickRoom: battle.field?.pseudoWeather?.trickroom?.duration || 0,
        sides,
        ended: battle.ended,
        winner: battle.winner || '',
    };
}

/**
 * Run a single battle with a fixed seed and log everything.
 */
async function runBattle(battleIdx, seed) {
    return new Promise((resolve, reject) => {
        const rng = new PRNG(seed);
        const battle = new Battle({
            formatid: 'gen4randombattle',
            seed: seed,
        });

        const turnLog = [];
        const teams = [];

        // Generate teams
        const p1Team = Teams.generate('gen4randombattle', { seed: [seed[0], seed[1], seed[2], seed[3] + 1] });
        const p2Team = Teams.generate('gen4randombattle', { seed: [seed[0], seed[1], seed[2], seed[3] + 2] });

        teams.push(p1Team.map(p => ({
            species: p.species,
            ability: p.ability,
            item: p.item,
            moves: p.moves,
            level: p.level,
            evs: p.evs,
            ivs: p.ivs,
            nature: p.nature,
            gender: p.gender,
        })));
        teams.push(p2Team.map(p => ({
            species: p.species,
            ability: p.ability,
            item: p.item,
            moves: p.moves,
            level: p.level,
            evs: p.evs,
            ivs: p.ivs,
            nature: p.nature,
            gender: p.gender,
        })));

        battle.setPlayer('p1', { name: 'Bot1', team: Teams.pack(p1Team) });
        battle.setPlayer('p2', { name: 'Bot2', team: Teams.pack(p2Team) });

        // Log initial state
        turnLog.push({
            turn: 0,
            actions: null,
            state: extractBattleState(battle),
        });

        const maxTurns = 200;
        let turnCount = 0;

        function doTurn() {
            if (battle.ended || turnCount >= maxTurns) {
                const result = {
                    battle_idx: battleIdx,
                    seed: seed,
                    teams: teams,
                    turns: turnLog,
                    winner: battle.winner || 'none',
                    total_turns: battle.turn,
                };
                resolve(result);
                return;
            }

            // Choose random legal actions for both sides
            const choices = [];
            for (const side of battle.sides) {
                const request = side.activeRequest;
                if (!request || request.wait) {
                    choices.push(null);
                    continue;
                }

                let choice = null;
                if (request.forceSwitch) {
                    // Must switch — pick first alive
                    const alive = side.pokemon
                        .map((p, i) => (!p.fainted && !p.isActive) ? i + 1 : null)
                        .filter(x => x !== null);
                    if (alive.length > 0) {
                        const pick = alive[Math.floor(rng.random() * alive.length)];
                        choice = `switch ${pick}`;
                    } else {
                        choice = 'default';
                    }
                } else if (request.active) {
                    // Can move or switch
                    const moveChoices = [];
                    const switchChoices = [];

                    if (request.active[0]) {
                        const moves = request.active[0].moves || [];
                        for (let i = 0; i < moves.length; i++) {
                            if (!moves[i].disabled && moves[i].pp > 0) {
                                moveChoices.push(`move ${i + 1}`);
                            }
                        }
                    }

                    for (let i = 0; i < side.pokemon.length; i++) {
                        const p = side.pokemon[i];
                        if (!p.fainted && !p.isActive) {
                            switchChoices.push(`switch ${i + 1}`);
                        }
                    }

                    const allChoices = [...moveChoices, ...switchChoices];
                    if (allChoices.length > 0) {
                        // 70% move, 30% switch (like RandomPlayerAI)
                        if (moveChoices.length > 0 && (switchChoices.length === 0 || rng.random() < 0.7)) {
                            choice = moveChoices[Math.floor(rng.random() * moveChoices.length)];
                        } else if (switchChoices.length > 0) {
                            choice = switchChoices[Math.floor(rng.random() * switchChoices.length)];
                        } else {
                            choice = 'default';
                        }
                    } else {
                        choice = 'default';
                    }
                } else {
                    choice = 'default';
                }

                choices.push(choice);
            }

            // Log actions
            const actionLog = {
                p1: choices[0],
                p2: choices[1],
            };

            // Execute choices
            for (let i = 0; i < battle.sides.length; i++) {
                if (choices[i]) {
                    battle.choose(battle.sides[i].id, choices[i]);
                }
            }

            turnCount++;

            // Log state after turn
            turnLog.push({
                turn: battle.turn,
                actions: actionLog,
                state: extractBattleState(battle),
            });

            // Continue
            setImmediate(doTurn);
        }

        doTurn();
    });
}

async function main() {
    // Ensure output directory exists
    const outDir = path.dirname(OUTPUT_FILE);
    if (!fs.existsSync(outDir)) {
        fs.mkdirSync(outDir, { recursive: true });
    }

    const ws = fs.createWriteStream(OUTPUT_FILE);
    let completed = 0;
    let errors = 0;

    console.log(`Running ${N_BATTLES} Gen 4 random battles...`);

    for (let i = 0; i < N_BATTLES; i++) {
        const seed = [i * 4 + 1, i * 4 + 2, i * 4 + 3, i * 4 + 4];
        try {
            const result = await runBattle(i, seed);
            ws.write(JSON.stringify(result) + '\n');
            completed++;
            if ((i + 1) % 10 === 0) {
                console.log(`  ${i + 1}/${N_BATTLES} battles done (${result.total_turns} turns)`);
            }
        } catch (err) {
            console.error(`  Battle ${i} failed: ${err.message}`);
            errors++;
        }
    }

    ws.end();
    console.log(`\nDone! ${completed} battles logged, ${errors} errors.`);
    console.log(`Output: ${OUTPUT_FILE}`);
}

main().catch(console.error);
