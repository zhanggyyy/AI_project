"""Bayesian-style hidden-state sampling for Love Letter.

The belief model is deliberately lightweight: it turns a hidden-information-safe
Observation into a distribution over plausible opponent hands, hidden set-aside
cards, and deck multisets. The search agent can then average values over sampled
determinizations instead of reading the true hidden state.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from loveletter_ai.cards import CARD_SPECS, CardName
from loveletter_ai.players import PlayerId
from loveletter_ai.state import Observation


CardCounter = dict[CardName, int]


@dataclass(frozen=True, slots=True)
class HiddenStateSample:
    """One possible assignment of hidden information."""

    opponent_hands: dict[PlayerId, tuple[CardName, ...]]
    deck: tuple[CardName, ...]
    hidden_card: CardName | None


@dataclass(slots=True)
class BayesianBelief:
    """Approximate posterior over hidden cards given one player's observation."""

    viewer: PlayerId
    deck_size: int
    remaining_counts: CardCounter
    opponent_hand_sizes: dict[PlayerId, int]
    opponent_constraints: dict[PlayerId, set[CardName]]

    @classmethod
    def from_observation(
        cls,
        observation: Observation,
        possible_cards: dict[PlayerId, set[CardName]] | None = None,
    ) -> "BayesianBelief":
        """Build a belief from public state, own hand, and optional constraints."""

        remaining = full_deck_counts()
        for card in observation.own_hand:
            decrement(remaining, card)
        for player in observation.players:
            for card in player.discard_pile:
                decrement(remaining, card)

        opponent_hand_sizes: dict[PlayerId, int] = {}
        opponent_constraints: dict[PlayerId, set[CardName]] = {}
        possible_cards = possible_cards or {}

        for player in observation.players:
            if player.id == observation.viewer or player.eliminated:
                continue
            if player.hand_size <= 0:
                continue
            opponent_hand_sizes[player.id] = player.hand_size
            constrained = set(possible_cards.get(player.id, set(CardName)))
            opponent_constraints[player.id] = constrained or set(CardName)

        return cls(
            viewer=observation.viewer,
            deck_size=observation.deck_size,
            remaining_counts=remaining,
            opponent_hand_sizes=opponent_hand_sizes,
            opponent_constraints=opponent_constraints,
        )

    def opponent_hand_distribution(self, player_id: PlayerId) -> dict[CardName, float]:
        """Return a simple marginal P(card in opponent hand | observation)."""

        allowed = self.opponent_constraints.get(player_id, set(CardName))
        weighted = {
            card: count
            for card, count in self.remaining_counts.items()
            if count > 0 and card in allowed
        }
        return normalize_counts(weighted)

    def future_draw_distribution(self) -> dict[CardName, float]:
        """Return the current marginal distribution for an unknown future draw."""

        return normalize_counts(
            {card: count for card, count in self.remaining_counts.items() if count > 0}
        )

    def sample_hidden_state(self, rng: random.Random) -> HiddenStateSample:
        """Sample opponent hands, hidden card, and deck order without replacement."""

        counts = dict(self.remaining_counts)
        opponent_hands: dict[PlayerId, tuple[CardName, ...]] = {}

        for player_id, hand_size in sorted(
            self.opponent_hand_sizes.items(),
            key=lambda item: int(item[0]),
        ):
            hand = []
            for _ in range(hand_size):
                allowed = [
                    card
                    for card in self.opponent_constraints.get(player_id, set(CardName))
                    if counts.get(card, 0) > 0
                ]
                if not allowed:
                    allowed = [card for card, count in counts.items() if count > 0]
                if not allowed:
                    break
                chosen = weighted_card_choice(counts, allowed, rng)
                hand.append(chosen)
                decrement(counts, chosen)
            opponent_hands[player_id] = tuple(hand)

        hidden_card = None
        remaining_unknown = sum(counts.values())
        if remaining_unknown > self.deck_size:
            hidden_card = weighted_card_choice(
                counts,
                [card for card, count in counts.items() if count > 0],
                rng,
            )
            decrement(counts, hidden_card)

        deck = expand_counts(counts)
        rng.shuffle(deck)
        if len(deck) > self.deck_size:
            deck = deck[: self.deck_size]

        return HiddenStateSample(
            opponent_hands=opponent_hands,
            deck=tuple(deck),
            hidden_card=hidden_card,
        )

    def apply_sample_to_env(self, env, sample: HiddenStateSample) -> None:
        """Overwrite hidden fields in a cloned environment with a sample."""

        state = env.state
        if state is None:
            raise RuntimeError("Cannot apply a belief sample before reset.")

        for player_id, hand in sample.opponent_hands.items():
            if player_id in state.player_states:
                state.player_states[player_id].hand = list(hand)

        state.deck = list(sample.deck)
        state.hidden_card = sample.hidden_card


def full_deck_counts() -> CardCounter:
    return {card: spec.count for card, spec in CARD_SPECS.items()}


def decrement(counts: CardCounter, card: CardName) -> None:
    if counts.get(card, 0) <= 0:
        return
    counts[card] -= 1


def expand_counts(counts: CardCounter) -> list[CardName]:
    deck: list[CardName] = []
    for card, count in counts.items():
        deck.extend([card] * max(count, 0))
    return deck


def normalize_counts(counts: CardCounter | dict[CardName, int]) -> dict[CardName, float]:
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {card: count / total for card, count in counts.items() if count > 0}


def weighted_card_choice(
    counts: CardCounter,
    cards: list[CardName],
    rng: random.Random,
) -> CardName:
    weights = [counts[card] for card in cards]
    total = sum(weights)
    if total <= 0:
        raise ValueError("Cannot sample from an empty card distribution.")
    threshold = rng.uniform(0, total)
    running = 0.0
    for card, weight in zip(cards, weights):
        running += weight
        if threshold <= running:
            return card
    return cards[-1]


def card_counts(cards: list[CardName]) -> Counter[CardName]:
    return Counter(cards)
