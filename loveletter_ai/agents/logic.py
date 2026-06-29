"""Symbolic reasoning agents."""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from loveletter_ai.actions import Action
from loveletter_ai.cards import CardName
from loveletter_ai.players import PlayerId
from loveletter_ai.state import Observation, PublicPlayerView

from loveletter_ai.agents.heuristics import ImprovedHeuristicAgent
from loveletter_ai.agents.utils import card_from_label, card_value


@dataclass(slots=True)
class LogicKnowledgeBase:
    """Symbolic card-set constraints for hidden opponent hands."""

    viewer: PlayerId | None = None
    possible: dict[PlayerId, set[CardName]] = field(default_factory=dict)
    player_names: dict[str, PlayerId] = field(default_factory=dict)
    processed_events: int = 0
    deductions: list[str] = field(default_factory=list)
    pending_guard_guesses: list[tuple[PlayerId | None, CardName]] = field(
        default_factory=list
    )

    def update(self, observation: Observation) -> None:
        if self.viewer != observation.viewer:
            self.viewer = observation.viewer
            self.possible.clear()
            self.player_names.clear()
            self.processed_events = 0
            self.deductions.clear()
            self._deduce(f"Started knowledge base for P{int(observation.viewer)}.")

        self._ensure_players(observation.players, observation.viewer)
        self._process_new_events(observation)
        self._remove_visible_cards(observation)
        self._drop_eliminated_players(observation.players)

    def possible_cards(self, player_id: PlayerId) -> set[CardName]:
        return set(self.possible.get(player_id, set()))

    def debug_lines(self) -> list[str]:
        lines = ["Logic deductions"]
        if not self.deductions:
            lines.append("  --")
        else:
            lines.extend(f"  - {deduction}" for deduction in self.deductions[-12:])

        lines.append("Possible opponent hands")
        for player_id, cards in sorted(self.possible.items(), key=lambda item: int(item[0])):
            labels = ", ".join(card.value for card in sorted(cards, key=card_value))
            lines.append(f"  P{int(player_id)}: {labels or '--'}")
        return lines

    def note_action(self, action: Action) -> None:
        if action.card == CardName.GUARD and action.guess is not None:
            self.pending_guard_guesses.append((action.target, action.guess))
            self._deduce(f"Observed Guard guess: {action.guess.value}.")

    def _ensure_players(
        self,
        players: tuple[PublicPlayerView, ...],
        viewer: PlayerId,
    ) -> None:
        all_cards = set(CardName)
        for player in players:
            self.player_names[player.name] = player.id
            if player.id != viewer and player.id not in self.possible:
                self.possible[player.id] = set(all_cards)
                self._deduce(f"Initialized possible hand set for {player.name}.")

    def _remove_visible_cards(self, observation: Observation) -> None:
        visible_cards = set(observation.own_hand)
        for player in observation.players:
            visible_cards.update(player.discard_pile)

        for player_id, cards in self.possible.items():
            before = set(cards)
            cards.difference_update(visible_cards)
            removed = before - cards
            if removed:
                labels = ", ".join(card.value for card in sorted(removed, key=card_value))
                self._deduce(f"P{int(player_id)} cannot hold visible cards: {labels}.")

    def _process_new_events(self, observation: Observation) -> None:
        events = observation.visible_events[self.processed_events :]
        messages = [event.message for event in events]
        for index, message in enumerate(messages):
            self._apply_priest_observation(message)
            self._apply_countess_play(message)
            self._apply_king_swap(message)
            if "The guess was wrong." in message:
                previous = messages[index - 1] if index > 0 else ""
                self._apply_guard_failure(previous, message)
        self.processed_events = len(observation.visible_events)

    def _apply_priest_observation(self, message: str) -> None:
        match = re.search(r"saw (.+) holding (.+)\.", message)
        if not match:
            return
        target_name = match.group(1)
        card = card_from_label(match.group(2))
        target = self.player_names.get(target_name)
        if target is None or card is None:
            return
        self.possible[target] = {card}
        self._deduce(f"Priest observation proves {target_name} holds {card.value}.")

    def _apply_countess_play(self, message: str) -> None:
        match = re.search(r"^(.+) played Countess\.", message)
        if not match:
            return
        player_name = match.group(1)
        player_id = self.player_names.get(player_name)
        if player_id is None or player_id not in self.possible:
            return
        self.possible[player_id].discard(CardName.COUNTESS)
        self._deduce(
            f"{player_name} played Countess, so Countess is no longer in hand."
        )

    def _apply_king_swap(self, message: str) -> None:
        match = re.search(r"^(.+) swapped hands with (.+)\.", message)
        if not match:
            return

        first = self.player_names.get(match.group(1))
        second = self.player_names.get(match.group(2))
        all_cards = set(CardName)
        for player_id in (first, second):
            if player_id is not None and player_id in self.possible:
                self.possible[player_id] = set(all_cards)
                self._deduce(
                    f"P{int(player_id)} swapped hands, so prior hand facts were reset."
                )

    def _apply_guard_failure(self, previous: str, current: str) -> None:
        guess_match = re.search(r"guessed (.+)'s card\.", previous)
        wrong_match = re.search(r"The guess was wrong\. (.+) stays alive\.", current)
        if not guess_match or not wrong_match:
            return

        target_name = wrong_match.group(1)
        target = self.player_names.get(target_name)
        guessed_card = self._pop_pending_guard_guess(target)
        if target is None or guessed_card is None or target not in self.possible:
            return

        if guessed_card in self.possible[target]:
            self.possible[target].discard(guessed_card)
            self._deduce(
                f"Failed Guard guess proves {target_name} is not {guessed_card.value}."
            )

    def _pop_pending_guard_guess(self, target: PlayerId | None) -> CardName | None:
        for index, (pending_target, guess) in enumerate(self.pending_guard_guesses):
            if target is None or pending_target is None or pending_target == target:
                del self.pending_guard_guesses[index]
                return guess
        return None

    def _drop_eliminated_players(self, players: tuple[PublicPlayerView, ...]) -> None:
        eliminated = {player.id for player in players if player.eliminated}
        for player_id in list(self.possible):
            if player_id in eliminated:
                del self.possible[player_id]
                self._deduce(f"Removed eliminated P{int(player_id)} from hand tracking.")

    def _deduce(self, message: str) -> None:
        if not self.deductions or self.deductions[-1] != message:
            self.deductions.append(message)


class LogicAgent:
    """Symbolic reasoning policy backed by a persistent knowledge base."""

    def __init__(self, name: str = "LogicAgent", debug: bool = False) -> None:
        self.name = name
        self.debug = debug
        self.kb = LogicKnowledgeBase()
        self.fallback = ImprovedHeuristicAgent(name=f"{name}Fallback")

    def choose_action(
        self,
        observation: Observation,
        legal_actions: Sequence[Action],
        rng: random.Random,
    ) -> Action:
        if not legal_actions:
            raise ValueError("LogicAgent received no legal actions.")

        self.kb.update(observation)
        if self.debug:
            print("\n".join(self.kb.debug_lines()))

        logical = self._deterministic_guard_action(legal_actions)
        if logical is not None:
            return logical

        return self.fallback.choose_action(observation, legal_actions, rng)

    def observe_action(self, action: Action) -> None:
        self.kb.note_action(action)

    def _deterministic_guard_action(
        self,
        legal_actions: Sequence[Action],
    ) -> Action | None:
        for action in legal_actions:
            if action.card != CardName.GUARD or action.target is None:
                continue
            possible = self.kb.possible_cards(action.target)
            if len(possible) == 1 and action.guess in possible:
                return action
        return None
