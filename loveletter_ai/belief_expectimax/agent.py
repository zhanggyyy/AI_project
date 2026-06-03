"""Belief-state expectimax agent with explicit chance nodes."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence

from loveletter_ai.actions import Action
from loveletter_ai.agents import LogicKnowledgeBase, NaiveHeuristicAgent
from loveletter_ai.cards import CARD_SPECS, CardName
from loveletter_ai.players import PlayerId
from loveletter_ai.state import Observation, Phase

from loveletter_ai.belief_expectimax.belief import BayesianBelief


class BeliefExpectimaxAgent:
    """Expectimax over sampled Bayesian hidden states.

    Root chance nodes sample opponent hands and the unseen deck from the belief.
    Draw chance nodes enumerate possible future draws by card multiplicity.
    Opponent actions are treated as uniformly random chance nodes.
    """

    def __init__(
        self,
        name: str = "BeliefExpectimaxAgent",
        depth: int = 3,
        samples: int = 40,
        debug: bool = False,
    ) -> None:
        if depth < 1:
            raise ValueError("BeliefExpectimaxAgent depth must be at least 1.")
        if samples < 1:
            raise ValueError("BeliefExpectimaxAgent samples must be at least 1.")
        self.name = name
        self.depth = depth
        self.samples = samples
        self.debug = debug
        self.kb = LogicKnowledgeBase()
        self.fallback = NaiveHeuristicAgent(name=f"{name}Fallback")

    def choose_action(
        self,
        observation: Observation,
        legal_actions: Sequence[Action],
        rng: random.Random,
    ) -> Action:
        """Fallback policy for observation-only runners."""

        self.kb.update(observation)
        return self.fallback.choose_action(observation, legal_actions, rng)

    def choose_action_from_env(
        self,
        env,
        player_id: PlayerId,
        rng: random.Random,
    ) -> Action:
        """Choose an action by maximizing expected utility under the belief."""

        observation = env.observe(player_id)
        legal_actions = env.legal_actions(player_id)
        if not legal_actions:
            raise ValueError("BeliefExpectimaxAgent received no legal actions.")

        self.kb.update(observation)
        possible_cards = {
            opponent: self.kb.possible_cards(opponent)
            for opponent in self.kb.possible
        }
        belief = BayesianBelief.from_observation(observation, possible_cards)

        scored: list[tuple[float, int, Action]] = []
        for index, action in enumerate(legal_actions):
            value = self._expected_root_action_value(env, belief, action, player_id, rng)
            scored.append((value, index, action))

        best_value = max(value for value, _, _ in scored)
        best_actions = [
            (index, action)
            for value, index, action in scored
            if value == best_value
        ]

        if self.debug:
            self._print_debug(scored, belief)

        _, best_action = rng.choice(best_actions)
        return best_action

    def observe_action(self, action: Action) -> None:
        """Feed public action metadata into the persistent belief constraints."""

        self.kb.note_action(action)

    def _expected_root_action_value(
        self,
        env,
        belief: BayesianBelief,
        action: Action,
        root_player: PlayerId,
        rng: random.Random,
    ) -> float:
        total = 0.0
        for _ in range(self.samples):
            sample = belief.sample_hidden_state(rng)
            sampled_env = env.clone()
            belief.apply_sample_to_env(sampled_env, sample)
            total += self._expected_action_value(
                sampled_env,
                action,
                self.depth - 1,
                root_player,
            )
        return total / self.samples

    def _expectimax(self, env, depth: int, root_player: PlayerId) -> float:
        state = env.state
        if state is None:
            raise RuntimeError("Cannot search an environment before reset.")
        if state.phase == Phase.ROUND_OVER or depth <= 0:
            return self._evaluate(env, root_player)

        if state.phase == Phase.DRAW:
            return self._expected_draw_value(env, depth, root_player)

        actor = state.current_player
        if actor is None:
            return self._evaluate(env, root_player)

        legal_actions = env.legal_actions(actor)
        if not legal_actions:
            return self._evaluate(env, root_player)

        values = [
            self._expected_action_value(env, action, depth - 1, root_player)
            for action in legal_actions
        ]
        if actor == root_player:
            return max(values)
        return sum(values) / len(values)

    def _expected_action_value(
        self,
        env,
        action: Action,
        depth: int,
        root_player: PlayerId,
    ) -> float:
        successors = self._action_successors(env, action)
        return sum(
            probability * self._expectimax(child, depth, root_player)
            for probability, child in successors
        )

    def _expected_draw_value(self, env, depth: int, root_player: PlayerId) -> float:
        successors = self._draw_successors(env)
        if not successors:
            return self._evaluate(env, root_player)
        return sum(
            probability * self._expectimax(child, depth, root_player)
            for probability, child in successors
        )

    def _action_successors(self, env, action: Action) -> list[tuple[float, object]]:
        """Return stochastic successors for an action.

        Most card effects are deterministic once hidden state is sampled. Prince
        can trigger a random replacement draw, so it gets an explicit chance node.
        """

        if action.card != CardName.PRINCE:
            child = env.clone()
            child.step(action)
            return [(1.0, child)]

        state = env.state
        if state is None or action.target is None:
            child = env.clone()
            child.step(action)
            return [(1.0, child)]

        target_state = state.player_states[action.target]
        if not target_state.hand or target_state.hand[-1] == CardName.PRINCESS:
            child = env.clone()
            child.step(action)
            return [(1.0, child)]

        if not state.deck:
            child = env.clone()
            child.step(action)
            return [(1.0, child)]

        counts = Counter(state.deck)
        successors = []
        total = len(state.deck)
        for card, count in counts.items():
            child = env.clone()
            child_state = child.state
            if child_state is None:
                continue
            child_state.deck.remove(card)
            child_state.deck.append(card)
            child.step(action)
            successors.append((count / total, child))
        return successors

    def _draw_successors(self, env) -> list[tuple[float, object]]:
        state = env.state
        if state is None:
            raise RuntimeError("Cannot draw from an environment before reset.")
        if state.current_player is None:
            return []
        if not state.deck:
            child = env.clone()
            child.begin_action_phase()
            return [(1.0, child)]

        counts = Counter(state.deck)
        total = len(state.deck)
        successors = []
        for card, count in counts.items():
            child = env.clone()
            child_state = child.state
            if child_state is None:
                continue
            player_state = child_state.player_states[child_state.current_player]
            player_state.protected = False
            child_state.deck.remove(card)
            player_state.hand.append(card)
            child_state.phase = Phase.ACTION
            successors.append((count / total, child))
        return successors

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
        hand_value = max(
            (CARD_SPECS[card].value for card in player_state.hand),
            default=0,
        )
        score = hand_value * 12.0 + player_state.discard_total
        if player_state.protected:
            score += 10.0
        if CardName.PRINCESS in player_state.hand:
            score += 8.0
        return score

    def _print_debug(
        self,
        scored: list[tuple[float, int, Action]],
        belief: BayesianBelief,
    ) -> None:
        print(f"{self.name} belief marginals")
        for player_id in sorted(belief.opponent_hand_sizes, key=int):
            distribution = belief.opponent_hand_distribution(player_id)
            parts = [
                f"{card.value}:{probability:.2f}"
                for card, probability in sorted(
                    distribution.items(),
                    key=lambda item: CARD_SPECS[item[0]].value,
                )
            ]
            print(f"  P{int(player_id)} {' '.join(parts) or '--'}")
        for value, _, action in scored:
            print(f"{self.name} eval {action.label()}: {value:.2f}")
