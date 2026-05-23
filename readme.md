# Love Letter AI Environment

A multi-agent imperfect-information card game environment based on *Love Letter*.

This project is designed for AI research and CS188-style agent development, including:

- Rule-based agents
- Bayesian inference
- Expectimax
- Hidden-information reasoning
- Monte Carlo Tree Search (MCTS)
- Reinforcement Learning
- POMDP approximations

---

# Game Overview

Love Letter is a lightweight hidden-information card game for 2–4 players.

Players attempt to eliminate opponents or survive until the deck is exhausted while holding the highest-value card.

The game contains:
- stochastic card draws
- hidden information
- bluffing
- probabilistic reasoning
- opponent modeling

---

# Deck Composition

| Card | Value | Count |
|------|------|------|
| Guard | 1 | 5 |
| Priest | 2 | 2 |
| Baron | 3 | 2 |
| Handmaid | 4 | 2 |
| Prince | 5 | 2 |
| King | 6 | 1 |
| Countess | 7 | 1 |
| Princess | 8 | 1 |

Total cards: 16

---

# Game Setup

1. Shuffle all cards.
2. Remove one card face-down from the game.
3. Each player draws one starting card.
4. Choose a starting player randomly.

---

# Turn Structure

Each turn consists of:

## 1. Draw Phase

The player draws one card from the deck.

Hand size becomes 2.

---

## 2. Action Phase

The player chooses exactly one card to play.

The played card’s effect resolves immediately.

The remaining card stays in hand.

---

## 3. Elimination Check

If a player:
- discards the Princess
- loses a Baron comparison
- is correctly identified by Guard

they are eliminated immediately.

---

# Round End Conditions

A round ends when:

## Condition 1: Only one player remains alive

That player wins the round.

---

## Condition 2: Deck becomes empty

All surviving players reveal hands.

Winner:
- highest hand value
- tie-breaker: highest discard pile total

---

# Card Rules

---

## Guard (1)

Guess another player's card (not Guard).

If correct:
- target is eliminated.

---

## Priest (2)

Privately view another player's hand.

---

## Baron (3)

Compare hands privately.

Lower value player is eliminated.

---

## Handmaid (4)

Gain protection until your next turn.

Protected players cannot be targeted.

---

## Prince (5)

Choose any player (including self).

Target discards hand and redraws.

If Princess is discarded:
- target is eliminated.

If deck is empty:
- draw the set-aside hidden card.

---

## King (6)

Swap hands with another player.

---

## Countess (7)

Must be played if held together with:
- King
- or Prince

Otherwise has no effect.

---

## Princess (8)

If discarded for any reason:
- immediate elimination.

---

# Information Structure

The game contains both:
- public information
- private information

## Public Information

- played cards
- eliminated players
- protection status
- deck size

## Private Information

- hands
- hidden set-aside card

---

# AI Environment Goals

This environment supports experimentation with:

- hidden-state reasoning
- belief tracking
- adversarial search
- stochastic planning
- bluff detection

---

# Suggested Agent Baselines

## Random Agent

Selects legal moves uniformly at random.

---

## Rule-Based Agent

Uses handcrafted heuristics:
- Guard guessing
- Princess protection
- Countess inference
- card counting

---

## Bayesian Agent

Maintains probability distributions over opponent hands.

---

## MCTS Agent

Uses Monte Carlo rollouts over hidden states.

---

# State Representation

Suggested GameState fields:

```python
class GameState:
    players
    deck
    discard_piles
    current_player
    protected_players
    eliminated_players
    hidden_card