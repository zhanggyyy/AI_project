# CS188 Project Opening Presentation: Love Letter AI Environment

## 1. Topic

**Project Title:**  
Love Letter AI: Reasoning and Planning in an Imperfect-Information Card Game

This project builds a CS188-style AI environment for *Love Letter*, a small
hidden-information card game. The goal is not only to make a playable game, but
to use the game as a research testbed for agents that reason under uncertainty.

The project focuses on:

- deterministic game simulation
- legal action generation
- terminal-based visualization
- imperfect-information reasoning
- symbolic inference
- adversarial search
- future Bayesian and game-theoretic agents

## 2. Motivation

Many classic AI examples in introductory courses use perfect-information games,
such as Pacman search problems, gridworlds, minimax trees, and deterministic
planning environments. These are useful, but many real-world decision problems
are not fully observable.

Love Letter is a good project domain because it is small enough to implement
cleanly, but rich enough to contain important AI challenges:

- **Hidden information:** players cannot see opponents' hands or the hidden
  removed card.
- **Stochasticity:** the deck is shuffled and players draw unknown cards.
- **Adversarial decision-making:** every action can reveal information or
  eliminate another player.
- **Inference from public evidence:** discards, failed Guard guesses, Countess
  plays, and Priest observations all change what an agent should believe.
- **Strategic uncertainty:** a good agent must reason not only about the game
  state, but also about what other agents may know.

This makes Love Letter a natural bridge between CS188 topics such as search,
game trees, probability, Bayesian reasoning, and partially observable
environments.

## 3. Environment Design

The codebase is designed around a strict separation between the true game state
and each agent's observation.

### Core Objects

- `GameState`: the complete true state, including deck order, all hands, discard
  piles, protection status, eliminated players, and hidden card.
- `Observation`: the information visible to a particular player.
- `Action`: a declarative command such as playing Guard, selecting a target, and
  making a guess.
- `Environment`: owns all mutation, legality checks, card effects, and
  deterministic randomness.
- `Agent`: receives an observation and a list of legal actions, then chooses one
  action.

### Design Principles

1. **The environment enforces legality.**  
   Agents never directly mutate game state.

2. **Hidden information is preserved.**  
   Agents receive observations, not the full `GameState`.

3. **The game is reproducible.**  
   The environment uses an explicit random seed for shuffling, starting-player
   selection, and agent tie-breaking.

4. **The system is extensible.**  
   The same environment can support Random agents, heuristic agents, logic
   agents, Expectimax, MCTS, Bayesian filtering, and future reinforcement
   learning.

## 4. Implemented Agents

### 4.1 RandomAgent

**Algorithm**

The RandomAgent is the simplest baseline:

1. Receive the legal action list from the environment.
2. Select one legal action uniformly at random.
3. Return that action.

Pseudo-code:

```text
function choose_action(observation, legal_actions):
    return UniformRandom(legal_actions)
```

**Purpose**

RandomAgent is not intelligent, but it is important as a baseline. It helps us
measure whether more advanced agents actually improve performance.

**Relevance to CS188**

RandomAgent connects to:

- stochastic policies
- baseline evaluation
- empirical comparison of agent performance
- randomized decision-making

In CS188 terms, it is useful as a control group when evaluating search-based or
reasoning-based agents.

### 4.2 NaiveHeuristicAgent

**Algorithm**

The NaiveHeuristicAgent uses simple handcrafted rules. It does not maintain a
persistent belief state.

Main heuristics:

- avoid discarding Princess
- prefer Handmaid when holding Princess
- prefer Guard guesses, especially high-impact cards
- avoid risky Baron plays when the remaining card is weak
- otherwise prefer actions that preserve a stronger hand

Pseudo-code:

```text
function choose_action(observation, legal_actions):
    best_score = -infinity
    best_actions = []

    for action in legal_actions:
        score = heuristic_score(action, own_hand)

        if action discards Princess:
            score -= very_large_penalty

        if holding Princess and action is Handmaid:
            score += large_bonus

        if action is Guard:
            score += guard_bonus

        if action is Baron and remaining hand is weak:
            score -= risk_penalty

        update best_actions using score

    return random tie-break among best_actions
```

**Purpose**

This agent represents a human-like beginner strategy. It does not perform deep
search, but it captures obvious rules of survival.

**Relevance to CS188**

NaiveHeuristicAgent connects to:

- evaluation functions
- reflex agents
- feature-based decision-making
- heuristic design

This is similar to the reflex agents in Pacman: the agent does not plan far into
the future, but it uses a scoring function over legal actions.

### 4.3 LogicAgent

**Algorithm**

LogicAgent separates inference from action selection.

It contains:

- `LogicKnowledgeBase`: symbolic inference over possible opponent cards
- action policy: use certain knowledge when available, otherwise fall back to
  NaiveHeuristicAgent

The knowledge base maintains a set of possible cards for each opponent.

It updates from:

- public discard history
- own hand
- failed Guard guesses
- Priest observations
- Countess plays
- King hand swaps
- eliminated players

Example:

```text
If Alice guesses Bob has Princess using Guard,
and the guess fails,
then Bob cannot currently have Princess.
```

Pseudo-code:

```text
function update_kb(observation):
    remove own hand and all discard piles from opponent possible sets

    for each new event:
        if Guard guess failed:
            remove guessed card from target's possible set

        if Priest observation:
            set target's possible cards to observed card

        if Countess was played:
            remove Countess from that player's current hand set

        if King swapped hands:
            reset stale hand facts for affected players

function choose_action(observation, legal_actions):
    update_kb(observation)

    if a Guard action can eliminate a target with known card:
        return that Guard action

    return NaiveHeuristicAgent.choose_action(...)
```

**Purpose**

LogicAgent is the first true reasoning agent. It does not assign probabilities.
It only reasons about whether a card is possible or impossible.

**Relevance to CS188**

LogicAgent connects directly to:

- propositional logic
- knowledge bases
- logical inference
- constraint propagation
- reasoning under partial observability

It is similar in spirit to CS188 logic planning and knowledge-based agents:
given observations and rules, the agent derives facts that were not directly
observed.

### 4.4 ExpectimaxAgent

**Algorithm**

The current ExpectimaxAgent is a first search-based baseline. It performs
depth-limited expectimax over cloned environment states.

At each node:

- if it is the agent's turn, choose the maximum-value action
- if it is the opponent's turn, average over the opponent's legal actions
- if the search reaches the depth limit, evaluate the state using a heuristic
  evaluation function

Pseudo-code:

```text
function expectimax(state, depth, root_player):
    if terminal(state) or depth == 0:
        return evaluation(state, root_player)

    if current_player == root_player:
        return max(expectimax(successor, depth - 1))

    else:
        return average(expectimax(successor, depth - 1))
```

The current version is a **determinized baseline**: it searches one concrete
environment state. This is useful for validating the search framework, but it is
not yet a full hidden-information Expectimax agent.

**Purpose**

ExpectimaxAgent introduces planning. Unlike RandomAgent or NaiveHeuristicAgent,
it evaluates possible future game states.

**Relevance to CS188**

ExpectimaxAgent connects directly to:

- adversarial search
- game trees
- stochastic opponents
- utility functions
- depth-limited planning
- evaluation functions

This is closely related to the CS188 Expectimax material used for games where
other agents may behave stochastically.

## 5. Future Agent Design

### 5.1 Hidden-Information Expectimax

The current ExpectimaxAgent searches a single known environment. However, in
Love Letter, the agent should not know the opponent's hand or the hidden card.

The next version will use **sampled states**:

1. Use `LogicKnowledgeBase` to infer which opponent cards are possible.
2. Generate multiple complete hidden states consistent with that knowledge.
3. Run Expectimax on each sampled state.
4. Average the value of each root action across samples.
5. Choose the action with the best expected value.

Pseudo-code:

```text
function hidden_information_expectimax(observation, kb):
    sampled_states = sample_states_consistent_with(kb)

    for action in legal_actions:
        values = []

        for state in sampled_states:
            value = expectimax(apply(action, state), depth)
            values.append(value)

        action_value[action] = average(values)

    return argmax(action_value)
```

This approach no longer assumes a single true state during planning. Instead, it
plans over multiple possible worlds.

**Connection to MCTS**

This is related to Monte Carlo Tree Search because both methods estimate action
quality by sampling possible futures. In this project, sampled hidden states
would provide a bridge from symbolic inference to Monte Carlo planning.

### 5.2 From Set to Distribution: Bayesian Filtering

The current LogicKnowledgeBase stores only possible/impossible facts:

```text
Bob may have: {Prince, King, Countess}
Bob cannot have: {Guard, Priest, Princess}
```

Future Bayesian reasoning will replace sets with probability distributions:

```text
P(Bob = Prince) = 0.45
P(Bob = King) = 0.35
P(Bob = Countess) = 0.20
```

This matters because not all evidence is deterministic.

Example:

- If a player plays Countess, logic alone does not prove they held King or Prince.
- However, probabilistically, a Countess play may increase the probability that
  they were forced to play it because they also had King or Prince.

Bayesian filtering would allow the agent to update beliefs after every public
and private event.

**Relevance to CS188**

This connects to:

- Bayes' rule
- belief states
- probabilistic inference
- hidden Markov models
- POMDP-style planning

The project can therefore evolve from pure logic to probabilistic reasoning.

### 5.3 Poker-Inspired Innovation: CFR and DeepStack

The most ambitious future direction is to borrow ideas from poker AI.

Poker and Love Letter share a central challenge: players make decisions with
private information, public history, and uncertainty about the opponent's
knowledge.

A useful reference is **DeepStack: Expert-Level Artificial Intelligence in
No-Limit Poker** by Moravcik et al. DeepStack introduced an approach for
imperfect-information poker that combines recursive reasoning, decomposition of
the game, and learned value estimation from self-play. The paper reports that
DeepStack defeated professional heads-up no-limit Texas hold'em players with
statistical significance.

DeepStack is relevant because it does not simply assume that the opponent acts
randomly. Instead, it reasons recursively in an imperfect-information game.

For this Love Letter project, a poker-inspired future agent could use:

- **continual re-solving:** recompute strategy at each decision point using the
  current public history and belief state
- **counterfactual values:** evaluate what each player could have achieved with
  different hidden cards or actions
- **CFR-style regret minimization:** learn strategies that are less exploitable
  than simple heuristic or random-opponent models
- **belief distributions:** replace hard possible-card sets with probabilistic
  ranges over opponent hands

In simple terms:

```text
Current Expectimax:
    assumes opponent actions are uniformly random

Future CFR-inspired agent:
    assumes opponent is also strategic
    updates belief ranges after every action
    learns low-exploitability policies through self-play
```

This would move the project from "planning against a random opponent" toward
"game-theoretic reasoning against strategic opponents."

## 6. Why This Project Fits CS188

This project combines multiple major CS188 themes in one environment.

| CS188 Topic | Project Connection |
|---|---|
| Reflex Agents | NaiveHeuristicAgent chooses actions using a heuristic score. |
| Search | ExpectimaxAgent searches future game states. |
| Adversarial Search | Players compete directly and can eliminate each other. |
| Stochasticity | Deck shuffling and card draws introduce randomness. |
| Logic | LogicAgent uses symbolic constraints over hidden cards. |
| Probability | Future Bayesian agent will maintain belief distributions. |
| POMDPs | Agents must act from partial observations. |
| MCTS | Future hidden-state sampling supports Monte Carlo planning. |
| Reinforcement Learning | The environment can support self-play training later. |

The project is especially relevant because it starts with course-level
algorithms and then naturally extends toward modern research topics in
imperfect-information games.

## 7. Current Progress

Already implemented:

- deterministic Love Letter environment
- legal action generation
- terminal visualization
- interactive human player mode
- RandomAgent
- NaiveHeuristicAgent
- LogicAgent with symbolic knowledge base
- determinized ExpectimaxAgent
- reproducible tests with fixed random seeds

Planned next steps:

1. Add evaluation metrics for agent tournaments.
2. Implement hidden-information Expectimax using sampled states.
3. Convert `LogicKnowledgeBase` from sets to Bayesian distributions.
4. Add self-play experiments.
5. Explore CFR-inspired agents for lower exploitability.

## 8. Expected Contribution

The expected contribution is a clean and extensible AI environment for studying
imperfect-information reasoning.

The project begins with simple CS188-style agents, then grows toward more
advanced research ideas:

- symbolic reasoning
- probabilistic belief tracking
- stochastic planning
- Monte Carlo sampling
- game-theoretic self-play

Love Letter is small, but the AI problems it creates are real. This makes it a
good domain for a CS188 project: understandable enough to implement, but deep
enough to demonstrate meaningful AI reasoning.

## 9. References

- Moravcik, M., Schmid, M., Burch, N., et al. "DeepStack: Expert-Level
  Artificial Intelligence in No-Limit Poker." *Science*, 356(6337), 508-513,
  2017. DOI: 10.1126/science.aam6960.  
  Source: https://pubmed.ncbi.nlm.nih.gov/28254783/

- CS188 course themes: search, adversarial search, logic, probabilistic
  inference, MDPs/POMDPs, and reinforcement learning.
