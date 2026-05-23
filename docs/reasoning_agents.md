# Reasoning Agent Architecture

The first agent layer separates policy choice from hidden-information inference.
This keeps the environment deterministic and makes future Bayesian or MCTS agents
easier to add.

## RandomAgent

`RandomAgent` is the baseline policy. It receives legal actions from the
environment and samples uniformly with the supplied RNG. It does not inspect card
semantics.

## NaiveHeuristicAgent

`NaiveHeuristicAgent` is stateless. It scores the legal action list with simple
handcrafted rules:

- never voluntarily discard Princess unless forced
- prefer Handmaid when holding Princess
- mildly prefer Guard guesses, especially high-impact cards
- avoid Baron when the remaining hand card is low
- otherwise prefer higher card values

The environment still enforces legality. The heuristic agent only ranks legal
actions.

## LogicAgent

`LogicAgent` combines two pieces:

- `LogicKnowledgeBase`: symbolic inference over possible opponent hand sets
- action policy: deterministic Guard if the KB proves a target's card, otherwise
  fallback to `NaiveHeuristicAgent`

The KB stores possible current cards for each opponent. It does not assign
probabilities.

## Logical Updates

The KB updates from observations and public/private events:

- own hand and all discard piles are visible, so those cards are impossible as
  current opponent hand cards
- failed Guard guesses remove the guessed card from the target's possible set
- Priest observations collapse the target's possible set to the observed card
- Countess plays remove Countess from that player's current hand possibilities
- eliminated players are removed from active hand tracking

This is deliberately conservative. For example, a Countess play can suggest that
the player may have held King or Prince, but because Countess can also be played
voluntarily, the KB does not treat that as a hard current-hand fact.

## Debug Output

`LogicAgent(debug=True)` prints:

- recent deductions
- possible opponent hand sets

This debug output is for terminal research and tests. Later Bayesian agents can
reuse the same event parsing but replace card sets with distributions.

## ExpectimaxAgent

`ExpectimaxAgent` is the first search-based baseline. It is currently a
determinized expectimax agent:

- it clones the concrete environment state
- root-player decision nodes take the maximum child value
- opponent decision nodes average over legal actions uniformly
- draw phases are advanced deterministically according to the cloned deck order
- leaf states are scored by a simple heuristic evaluation function

This version is intentionally not probabilistic hidden-information reasoning yet.
It is a clean control-flow scaffold for future work:

1. Replace concrete-state cloning with sampled hidden-state determinizations.
2. Use `LogicKnowledgeBase` to restrict sampled opponent hands.
3. Average over sampled states before choosing a root action.
4. Later, replace uniform opponent actions with learned or Bayesian opponent
   models.

The normal observation-only `choose_action` method falls back to
`NaiveHeuristicAgent`, because an observation by itself cannot legally simulate
future transitions. Runners that want expectimax search should call
`choose_action_from_env(env, player_id, rng)`.
