# Architecture Plan

This project is a CS188-style research environment for Love Letter. The first
priority is not to make a complete game, but to make the boundaries clean enough
that future agents can reason about hidden information without depending on
engine internals.

## Design Goals

- Keep the game engine deterministic under an explicit random seed.
- Preserve hidden information by separating true state from agent observations.
- Make legality enforcement the environment's responsibility.
- Keep agents simple: they receive an observation and legal actions, then choose.
- Support future Bayesian inference, MCTS rollouts, self-play, and RL wrappers.
- Provide terminal-first rendering for debugging and replay.

## Package Layout

```text
loveletter_ai/
  cards.py          Card identities and immutable card specs.
  actions.py        Action representations chosen by players.
  players.py        Player identifiers and public/private player data.
  state.py          True GameState, observations, events, and history records.
  agents.py         Agent protocol and baseline agent shells.
  environment.py    Deterministic environment boundary and legality checks.
  renderers/
    terminal.py     ASCII/text state and replay renderer.

docs/
  architecture.md   System design and design decisions.
  visualization.md  Terminal renderer contract and debug display plan.
```

## Core Abstractions

### Card

`CardSpec` describes a card type: name, value, count, and a short rules text.
Card specs are immutable and global. Individual physical cards do not need unique
IDs for the first version because Love Letter cards of the same rank are
exchangeable for game logic and belief tracking. If future experiments need
card-instance tracing, an `InstanceCard` wrapper can be added without changing
agent action APIs.

### Action

`Action` is an immutable command chosen by a player. It includes:

- `actor`: player taking the turn
- `card`: card being played
- `target`: optional target player
- `guess`: optional Guard guess

The action is intentionally declarative. The environment interprets and resolves
it, which keeps agents from mutating state or implementing rules themselves.

### Player

`Player` stores stable identity and display name. Runtime per-round data belongs
in `PlayerState`: hand, discard pile, eliminated status, and protection status.
This separation makes it easier to reset rounds and run many self-play episodes.

### GameState

`GameState` is the full truth and is owned by the environment. It includes deck,
hidden set-aside card, player states, active player, phase, event history, and a
seeded RNG handle or seed metadata.

Agents should not receive `GameState` directly. They receive an `Observation`,
which contains public information plus private information visible to that
agent. This preserves hidden information by construction.

### Environment

`LoveLetterEnv` is the only object that mutates state. Its duties are:

- initialize deterministic rounds from a seed
- produce legal actions
- validate chosen actions
- apply game rules
- emit observations and public/private events
- expose replay history

The initial skeleton defines this boundary but does not implement full card
effects yet.

### Agent

An `Agent` implements:

```python
choose_action(observation, legal_actions, rng) -> Action
```

The environment supplies already-filtered legal actions. Agents may use the RNG
for deterministic tie-breaking. Future agents can add richer internal state, but
the minimal API remains compatible with random, rule-based, Bayesian, MCTS, and
RL-style policies.

## Determinism

The environment owns a `random.Random` instance created from a seed. All shuffles,
starting-player choices, stochastic rollouts, and random agent tie-breakers should
flow from explicit RNG objects. Tests should never depend on module-level random
state.

## Hidden Information

Hidden information is protected through three layers:

1. `GameState` is not passed to agents.
2. `Observation` is created per viewer and reveals only that viewer's hand plus
   public information.
3. The terminal renderer accepts an optional `viewer` argument. Without a viewer,
   private hands and the hidden card are masked.

Debug tools may render perfect information, but that path should require an
explicit `reveal=True` flag.

## Legality

Agents only choose from legal actions, but the environment still validates every
submitted action. This gives two useful failure modes:

- agent bugs are caught immediately
- future UI or replay inputs cannot bypass rules

The Countess forced-play rule, protection targeting, Guard guess restrictions,
and target availability all belong in the environment legality layer.

## Future Extensions

- Bayesian agents can consume observations and public event histories.
- MCTS agents can sample determinizations from belief states and call cloned
  environments.
- RL wrappers can expose observations, action masks, and rewards without
  modifying the engine.
- Replay tools can serialize `HistoryEntry` records for deterministic debugging.
