"""Determinized expectimax search agent."""

from __future__ import annotations

import random
from collections.abc import Sequence

from loveletter_ai.actions import Action
from loveletter_ai.cards import CardName
from loveletter_ai.players import PlayerId
from loveletter_ai.state import Observation, Phase

from loveletter_ai.agents.heuristics import ImprovedHeuristicAgent
from loveletter_ai.agents.utils import card_value


class ExpectimaxAgent:
    """Depth-limited expectimax over a cloned deterministic environment.

    Observation-only calls fall back to a heuristic policy. Search runners can
    call ``choose_action_from_env`` when they intentionally expose a cloned true
    state for algorithmic comparison.
    """

    def __init__(
        self,
        name: str = "ExpectimaxAgent",
        depth: int = 3,
        debug: bool = False,
    ) -> None:
        if depth < 1:
            raise ValueError("ExpectimaxAgent depth must be at least 1.")
        self.name = name
        self.depth = depth
        self.debug = debug
        self.fallback = ImprovedHeuristicAgent(name=f"{name}Fallback")

    def choose_action(
        self,
        observation: Observation,
        legal_actions: Sequence[Action],
        rng: random.Random,
    ) -> Action:
        return self.fallback.choose_action(observation, legal_actions, rng)

    def choose_action_from_env(
        self,
        env,
        player_id: PlayerId,
        rng: random.Random,
    ) -> Action:
        legal_actions = env.legal_actions(player_id)
        if not legal_actions:
            raise ValueError("ExpectimaxAgent received no legal actions.")

        scored: list[tuple[float, int, Action]] = []
        for index, action in enumerate(legal_actions):
            child = env.clone()
            child.step(action)
            value = self._expectimax(child, self.depth - 1, player_id)
            scored.append((value, index, action))

        best_value = max(value for value, _, _ in scored)
        best_actions = [
            (index, action)
            for value, index, action in scored
            if value == best_value
        ]
        if self.debug:
            for value, _, action in scored:
                print(f"{self.name} eval {action.label()}: {value:.2f}")
        _, best_action = rng.choice(best_actions)
        return best_action

    def _expectimax(self, env, depth: int, root_player: PlayerId) -> float:
        state = env.state
        if state is None:
            raise RuntimeError("Cannot search an environment before reset.")
        if state.phase == Phase.ROUND_OVER or depth <= 0:
            return self._evaluate(env, root_player)

        if state.phase == Phase.DRAW:
            child = env.clone()
            child.begin_action_phase()
            return self._expectimax(child, depth, root_player)

        actor = state.current_player
        if actor is None:
            return self._evaluate(env, root_player)
        legal_actions = env.legal_actions(actor)
        if not legal_actions:
            return self._evaluate(env, root_player)

        values = []
        for action in legal_actions:
            child = env.clone()
            child.step(action)
            values.append(self._expectimax(child, depth - 1, root_player))

        if actor == root_player:
            return max(values)
        return sum(values) / len(values)

    def _evaluate(self, env, root_player: PlayerId) -> float:
        state = env.state
        if state is None:
            raise RuntimeError("Cannot evaluate an environment before reset.")

        if state.phase == Phase.ROUND_OVER:
            if state.winner == root_player:
                return 1_000.0
            if state.winner is None:
                return 0.0
            return -1_000.0

        root_state = state.player_states[root_player]
        if root_state.eliminated:
            return -900.0

        score = self._player_position_value(root_state)
        for player in state.players:
            if player.id == root_player:
                continue
            opponent_state = state.player_states[player.id]
            if opponent_state.eliminated:
                score += 120.0
            else:
                score -= 0.7 * self._player_position_value(opponent_state)
        return score

    def _player_position_value(self, player_state) -> float:
        hand_value = max((card_value(card) for card in player_state.hand), default=0)
        score = hand_value * 12.0 + player_state.discard_total
        if player_state.protected:
            score += 10.0
        if CardName.PRINCESS in player_state.hand:
            score += 8.0
        return score
