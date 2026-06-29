"""Random and heuristic policies."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence

from loveletter_ai.actions import Action
from loveletter_ai.cards import CARD_SPECS, CardName
from loveletter_ai.state import Observation

from loveletter_ai.agents.utils import (
    card_value,
    remaining_after_play,
    vulnerable_opponents,
)


class RandomAgent:
    """Baseline agent that samples uniformly from legal actions."""

    def __init__(self, name: str = "RandomAgent") -> None:
        self.name = name

    def choose_action(
        self,
        observation: Observation,
        legal_actions: Sequence[Action],
        rng: random.Random,
    ) -> Action:
        if not legal_actions:
            raise ValueError("RandomAgent received no legal actions.")
        return rng.choice(list(legal_actions))


class NaiveHeuristicAgent:
    """Simple score-based policy with no persistent hidden-state model."""

    def __init__(self, name: str = "NaiveHeuristicAgent") -> None:
        self.name = name

    def choose_action(
        self,
        observation: Observation,
        legal_actions: Sequence[Action],
        rng: random.Random,
    ) -> Action:
        if not legal_actions:
            raise ValueError("NaiveHeuristicAgent received no legal actions.")

        scored = [
            (self._score_action(observation, action), index, action)
            for index, action in enumerate(legal_actions)
        ]
        best_score = max(score for score, _, _ in scored)
        best_actions = [
            (index, action)
            for score, index, action in scored
            if score == best_score
        ]
        _, action = rng.choice(best_actions)
        return action

    def _score_action(self, observation: Observation, action: Action) -> int:
        score = 0
        hand = list(observation.own_hand)

        if action.card == CardName.PRINCESS:
            score -= 10_000

        if CardName.PRINCESS in hand and action.card == CardName.HANDMAID:
            score += 1_000

        if action.card == CardName.GUARD:
            score += 20
            if action.guess == CardName.PRINCESS:
                score += 8
            elif action.guess in (CardName.COUNTESS, CardName.KING, CardName.PRINCE):
                score += 4

        if action.card == CardName.BARON:
            remaining = remaining_after_play(hand, action.card)
            remaining_value = card_value(remaining[0]) if remaining else 0
            if remaining_value <= 4:
                score -= 80
            else:
                score += remaining_value * 5

        if action.card == CardName.HANDMAID:
            score += 30

        score += card_value(action.card)
        return score


class GreedyAgent:
    """Greedy one-step policy with hidden-card probability awareness."""

    def __init__(self, name: str = "GreedyAgent") -> None:
        self.name = name
        self.scorer = ImprovedHeuristicAgent(name=f"{name}Scorer")

    def choose_action(
        self,
        observation: Observation,
        legal_actions: Sequence[Action],
        rng: random.Random,
    ) -> Action:
        if not legal_actions:
            raise ValueError("GreedyAgent received no legal actions.")

        hand = set(observation.own_hand)

        if CardName.COUNTESS in hand and (
            CardName.KING in hand or CardName.PRINCE in hand
        ):
            countess = first_action_with_card(legal_actions, CardName.COUNTESS)
            if countess is not None:
                return countess

        scored = [
            (self.scorer.score_action(observation, action), index, action)
            for index, action in enumerate(legal_actions)
        ]
        best_score = max(score for score, _, _ in scored)
        best_actions = [
            (index, action)
            for score, index, action in scored
            if score == best_score
        ]
        _, action = rng.choice(best_actions)
        return action


class ImprovedHeuristicAgent:
    """Score-based heuristic using simple card-count beliefs."""

    def __init__(self, name: str = "ImprovedHeuristicAgent") -> None:
        self.name = name

    def choose_action(
        self,
        observation: Observation,
        legal_actions: Sequence[Action],
        rng: random.Random,
    ) -> Action:
        if not legal_actions:
            raise ValueError("ImprovedHeuristicAgent received no legal actions.")

        hand = list(observation.own_hand)
        if CardName.COUNTESS in hand and (
            CardName.KING in hand or CardName.PRINCE in hand
        ):
            countess = first_action_with_card(legal_actions, CardName.COUNTESS)
            if countess is not None:
                return countess

        scored = [
            (self.score_action(observation, action), index, action)
            for index, action in enumerate(legal_actions)
        ]
        best_score = max(score for score, _, _ in scored)
        best_options = [
            (index, action)
            for score, index, action in scored
            if score == best_score
        ]
        _, action = rng.choice(best_options)
        return action

    def score_action(self, observation: Observation, action: Action) -> float:
        hand = list(observation.own_hand)
        vulnerable = vulnerable_opponents(observation)
        remaining = remaining_after_play(hand, action.card)
        retained_value = max((card_value(card) for card in remaining), default=0)
        remaining_counts = unseen_card_counts(observation)
        unseen_total = sum(remaining_counts.values())
        expected_opponent_value = expected_card_value(remaining_counts)
        score = retained_value * 12.0 - card_value(action.card)

        if CardName.PRINCESS in hand:
            if action.card == CardName.HANDMAID:
                score += 3_000.0
            if action.card == CardName.PRINCESS:
                return -1_000_000

        if action.card == CardName.GUARD:
            score += 18.0
            if action.target in vulnerable:
                score += 18.0
            if action.guess is None:
                score -= 40.0
            else:
                guess_count = remaining_counts.get(action.guess, 0)
                if guess_count <= 0:
                    score -= 120.0
                else:
                    score += guard_guess_value(action.guess, guess_count, unseen_total)
        elif action.card == CardName.BARON:
            if action.target in vulnerable:
                if retained_value <= 3:
                    score -= 140.0
                else:
                    score += (retained_value - expected_opponent_value) * 24.0
            else:
                score -= 100.0
        elif action.card == CardName.PRIEST:
            score += 8.0 if action.target in vulnerable else 1.0
        elif action.card == CardName.PRINCE:
            if action.target == observation.viewer:
                if CardName.PRINCESS in remaining:
                    return -900_000
                score += 24.0 if retained_value <= 2 else -24.0
            elif action.target in vulnerable:
                princess_count = remaining_counts.get(CardName.PRINCESS, 0)
                princess_probability = (
                    princess_count / unseen_total if unseen_total else 0.0
                )
                score += 14.0 + princess_probability * 260.0
            else:
                score -= 40.0
        elif action.card == CardName.KING:
            if action.target in vulnerable:
                score += max(0.0, expected_opponent_value - retained_value) * 18.0
                if retained_value >= 6:
                    score -= 80.0
            else:
                score -= 80.0
        elif action.card == CardName.HANDMAID:
            score += 64.0

        return score

    def _score_action(self, observation: Observation, action: Action) -> float:
        return self.score_action(observation, action)


def actions_with_card(actions: Sequence[Action], card: CardName) -> list[Action]:
    return [action for action in actions if action.card == card]


def first_action_with_card(actions: Sequence[Action], card: CardName) -> Action | None:
    for action in actions:
        if action.card == card:
            return action
    return None


def targetable_actions(actions: Sequence[Action], targets: set) -> list[Action]:
    return [action for action in actions if action.target in targets]


def unseen_card_counts(observation: Observation) -> Counter[CardName]:
    counts = Counter({card: spec.count for card, spec in CARD_SPECS.items()})
    for card in observation.own_hand:
        counts[card] -= 1
    for player in observation.players:
        for card in player.discard_pile:
            counts[card] -= 1
    return Counter({card: max(0, count) for card, count in counts.items()})


def expected_card_value(counts: Counter[CardName]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return sum(card_value(card) * count for card, count in counts.items()) / total


def guard_guess_value(card: CardName, count: int, unseen_total: int) -> float:
    probability = count / unseen_total if unseen_total else 0.0
    return probability * 240.0 + card_value(card) * 4.0
