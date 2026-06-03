# Belief Expectimax Agent

This package implements a belief-state expectimax agent for the Love Letter
environment.

The agent has three layers:

1. `BayesianBelief` builds an approximate posterior from an `Observation`.
2. Root chance nodes sample opponent hands, the hidden card, and deck order.
3. Expectimax search maximizes the root player's value while averaging over
   opponent actions and future draw outcomes.

Use it with runners that expose the environment:

```python
from loveletter_ai.belief_expectimax import BeliefExpectimaxAgent

agent = BeliefExpectimaxAgent(depth=3, samples=40, debug=True)
action = agent.choose_action_from_env(env, player_id, env.rng)
```

The observation-only `choose_action` method falls back to the existing naive
heuristic, because full transition simulation requires an environment clone.
