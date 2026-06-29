
# RL Agent and MCTS Implementation Plan

This document describes how to add two next agents to the current Love Letter AI
project:

- `QLearningAgent`: a tabular reinforcement-learning baseline trained by
  self-play.
- `MCTSAgent`: a rollout-based search agent using cloned environments and
  sampled hidden-state determinizations.

The goal is to keep both agents compatible with the existing environment
boundary:

```python
choose_action(observation, legal_actions, rng) -> Action
choose_action_from_env(env, player_id, rng) -> Action
```

The normal `choose_action` API remains observation-safe. Search and training
code that needs simulation should use `choose_action_from_env`, matching the
current `ExpectimaxAgent` pattern.

## Current Project Hooks

The implementation should reuse these existing pieces:

- `LoveLetterEnv.clone()` for rollout/search branches.
- `LoveLetterEnv.legal_actions(player_id)` for action masks.
- `LoveLetterEnv.step(action)` and `begin_action_phase()` for transitions.
- `Observation` for public/private state visible to the acting player.
- `Action` as the canonical legal move object.
- `NaiveHeuristicAgent` as a fallback policy for rollouts and untrained states.

No algorithm should mutate `GameState` directly except through cloned
environments.

## Shared Utilities

Before implementing either agent, add a small helper layer in `agents.py` or a
new module such as `loveletter_ai/features.py`.

### Observation Features

`QLearningAgent` needs a stable, hashable state key. MCTS can also use the same
public feature shape for debugging and node labels.

Recommended state key:

```python
def observation_key(observation: Observation) -> tuple:
    return (
        observation.phase.value,
        int(observation.current_player) if observation.current_player is not None else None,
        observation.deck_size,
        tuple(sorted(card.value for card in observation.own_hand)),
        tuple(
            (
                int(player.id),
                player.hand_size,
                tuple(card.value for card in player.discard_pile),
                player.eliminated,
                player.protected,
            )
            for player in observation.players
        ),
    )
```

This is intentionally compact and imperfect-information-safe. It does not encode
opponent hidden cards or deck order.

### Action Keys

Q-values should not be keyed by raw `Action` objects alone, because the same
action shape should generalize across equivalent states.

Recommended action key:

```python
def action_key(action: Action) -> tuple:
    return (
        action.card.value,
        int(action.target) if action.target is not None else None,
        action.guess.value if action.guess is not None else None,
    )
```

When selecting an action, compute keys only from the current legal action list.
This preserves legality and avoids needing to enumerate all possible moves.

### Rewards

Use sparse terminal rewards first:

- `+1.0` if the learning player wins the round.
- `-1.0` if another player wins.
- `0.0` for non-terminal steps.

Optional shaping can be added after the baseline works:

- `+0.1` for eliminating an opponent.
- `-0.1` for being eliminated.
- `+0.02` for surviving a turn.

Keep the first implementation sparse so tests are easy to reason about.

## QLearningAgent

### Purpose

`QLearningAgent` should be the first RL baseline. It will learn a tabular
state-action value function over observation features. This is not a perfect
POMDP solution, but it is a useful project-level baseline and can be trained
quickly through self-play.

### Class Shape

Add this class to `loveletter_ai/agents.py` or split it into
`loveletter_ai/rl.py` and re-export it from `agents.py`.

```python
class QLearningAgent:
    def __init__(
        self,
        name: str = "QLearningAgent",
        alpha: float = 0.2,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        q_values: dict[tuple, float] | None = None,
    ) -> None:
        ...

    def choose_action(self, observation, legal_actions, rng) -> Action:
        ...

    def update(
        self,
        observation,
        action: Action,
        reward: float,
        next_observation,
        next_legal_actions,
    ) -> None:
        ...
```

The Q-table key should be:

```python
(observation_key(observation), action_key(action))
```

### Action Selection

Use epsilon-greedy action selection:

1. If `legal_actions` is empty, raise `ValueError`.
2. With probability `epsilon`, choose a random legal action.
3. Otherwise choose the legal action with the highest Q-value.
4. Break ties randomly with the provided `rng`.

Missing Q-values default to `0.0`.

### Update Rule

Use the standard tabular Q-learning update:

```python
old = Q[state_key, action_key]
target = reward + gamma * max_next_q
Q[state_key, action_key] = old + alpha * (target - old)
```

If the next state is terminal, `max_next_q = 0.0`.

### Training Loop

Add a training helper in a new file such as `loveletter_ai/training.py`.

Recommended function:

```python
def train_q_learning_self_play(
    episodes: int,
    seed: int | None = None,
    epsilon: float = 0.2,
) -> QLearningAgent:
    ...
```

Training flow for each episode:

1. Create `LoveLetterEnv(["Alice", "Bob"], seed=episode_seed)`.
2. Reset the environment.
3. Loop until `Phase.ROUND_OVER`.
4. Call `begin_action_phase()`.
5. Let the current player choose from observation and legal actions.
6. Store `(observation, action)` for that acting player.
7. Step the environment.
8. Give intermediate reward `0.0`.
9. At terminal state, update each player's last transition with win/loss reward.

For the first version, use one shared `QLearningAgent` instance for both seats.
This learns a seat-independent policy because the state/action keys include
player ids from the acting observation.

### Persistence

Add simple JSON save/load methods after the in-memory version works:

```python
agent.save_q_table(path)
QLearningAgent.load_q_table(path)
```

Because tuple keys are not directly JSON serializable, serialize keys with
`repr()` or convert them into nested lists. Prefer a small explicit conversion
helper over depending on pickle.

### Tests

Add tests in `test_agent/test_q_learning_agent.py`:

- epsilon `0.0` chooses the highest-valued legal action.
- equal Q-values break ties using the supplied RNG.
- `update()` moves a Q-value toward the expected target.
- training for a small number of episodes runs without illegal actions.
- saved and loaded Q-tables produce the same greedy action.

## MCTSAgent

### Purpose

`MCTSAgent` should search the current round by running many simulated games from
the current position. Love Letter has hidden information, so the search must not
use unknown opponent hands directly when acting from an observation. The
recommended first version is ISMCTS-style determinization:

1. Build several plausible complete states compatible with the root player's
   observation.
2. Run ordinary MCTS simulations on cloned deterministic environments.
3. Aggregate root action scores across determinizations.

The project already has a simpler concrete-state search pattern in
`ExpectimaxAgent`. MCTS should use the same `choose_action_from_env` entry point
for now, then add belief sampling once the rollout shell is stable.

### Class Shape

Add this class to `loveletter_ai/agents.py` or a new
`loveletter_ai/search.py`.

```python
class MCTSAgent:
    def __init__(
        self,
        name: str = "MCTSAgent",
        simulations: int = 200,
        exploration: float = 1.4,
        rollout_depth: int = 20,
        debug: bool = False,
    ) -> None:
        ...

    def choose_action(self, observation, legal_actions, rng) -> Action:
        ...

    def choose_action_from_env(self, env, player_id, rng) -> Action:
        ...
```

Like `ExpectimaxAgent`, plain `choose_action` should fall back to
`NaiveHeuristicAgent` because observation-only calls cannot safely simulate
unknown transitions yet.

### Node Data

Use a small dataclass:

```python
@dataclass
class MCTSNode:
    parent: MCTSNode | None
    action: Action | None
    player_to_move: PlayerId | None
    visits: int = 0
    value: float = 0.0
    children: dict[Action, MCTSNode] = field(default_factory=dict)
    untried_actions: list[Action] = field(default_factory=list)
```

`value` is always from the root player's perspective.

### UCT Selection

For a child node:

```python
uct = child.value / child.visits + exploration * sqrt(log(parent.visits) / child.visits)
```

Unvisited children should be selected before UCT calculation or treated as
infinite.

### Simulation Steps

Each MCTS simulation should do:

1. Clone the root environment.
2. Selection: descend through fully expanded nodes using UCT.
3. Expansion: choose one untried legal action and create a child.
4. Simulation: rollout with a simple policy until round end or `rollout_depth`.
5. Backpropagation: add the rollout result to each visited node.

The rollout policy can start as:

- root player: `NaiveHeuristicAgent`
- opponents: random legal actions

This gives more stable rollouts than pure random while keeping implementation
simple.

### Handling Draw Phase

MCTS nodes should represent decision points, not draw phases. During selection,
expansion, and rollout:

```python
while state.phase == Phase.DRAW:
    env.begin_action_phase()
```

If `begin_action_phase()` ends the round, return the terminal result.

### Rollout Evaluation

If rollout reaches `Phase.ROUND_OVER`:

- return `+1.0` when `state.winner == root_player`
- return `-1.0` when another player wins
- return `0.0` when no winner exists

If rollout hits `rollout_depth`, use the same heuristic as `ExpectimaxAgent` but
scaled into a small range, or reuse a helper:

```python
score = heuristic_position_value(env, root_player)
return max(-1.0, min(1.0, score / 100.0))
```

### Root Action Choice

After all simulations, choose the root legal action with the highest visit count.
Use average value as the tie-breaker:

```python
visits = child.visits
mean_value = child.value / child.visits
```

If no child was expanded, fall back to `NaiveHeuristicAgent`.

### Hidden Information Sampling

The first implementation can search the concrete cloned env, matching
`ExpectimaxAgent`. After that passes tests, add a determinization helper:

```python
def sample_determinization(env, root_player, rng) -> LoveLetterEnv:
    clone = env.clone()
    observation = clone.observe(root_player)
    ...
    return clone
```

Sampling requirements:

1. Keep the root player's actual hand unchanged.
2. Keep public discard piles unchanged.
3. Build the multiset of unseen cards from deck composition minus visible cards.
4. Randomly assign one hidden hand card to each alive opponent.
5. Randomly assign remaining cards to `deck` and `hidden_card`.
6. Preserve deck size from the observation.

Later, this sampler can use `LogicKnowledgeBase` to reject impossible opponent
hands, such as cards disproved by failed Guard guesses or revealed by Priest.

### Tests

Add tests in `test_agent/test_mcts_agent.py`:

- `choose_action_from_env()` returns an action in `env.legal_actions(actor)`.
- running with a fixed seed is deterministic.
- MCTS handles draw phase by advancing to action phase internally.
- rollout stops cleanly at terminal states.
- with `simulations=1`, the agent still returns a legal fallback or expanded
  action.
- sampled determinizations preserve the root player's hand and public discard
  piles.

## Suggested Implementation Order

1. Add shared `observation_key`, `action_key`, and reward/evaluation helpers.
2. Implement `QLearningAgent` action selection and update logic.
3. Add focused Q-learning unit tests.
4. Add a minimal self-play training helper.
5. Implement concrete-state `MCTSAgent` using `env.clone()`.
6. Add MCTS unit tests.
7. Add determinization sampling for hidden information.
8. Add an example runner for `QLearningAgent` and `MCTSAgent` under
   `test_agent/` or a new `examples/` folder.

## Expected Files To Change

Likely code files:

- `loveletter_ai/agents.py`
- `loveletter_ai/features.py`
- `loveletter_ai/training.py`
- `test_agent/test_q_learning_agent.py`
- `test_agent/test_mcts_agent.py`

Optional docs/examples:

- `docs/reasoning_agents.md`
- `test_agent/test_q_learning_agent.py`
- `test_agent/test_mcts_agent.py`

## Open Design Choices

These are worth confirming before coding:

- Whether Q-learning should support only two-player games first, or all 2-4
  player games immediately.
- Whether Q-learning persistence should use JSON or pickle.
- Whether MCTS should first match `ExpectimaxAgent` by using concrete true state,
  then add hidden-state sampling, or include determinization from the start.
- Whether rollout opponents should be random, heuristic, or configurable.
