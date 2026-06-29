"""Tabular reinforcement learning and Monte Carlo tree search agents."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from loveletter_ai.actions import Action
from loveletter_ai.cards import CardName
from loveletter_ai.players import PlayerId
from loveletter_ai.state import Observation, Phase

from loveletter_ai.agents.heuristics import (
    ImprovedHeuristicAgent,
    expected_card_value,
    unseen_card_counts,
)
from loveletter_ai.agents.logic import LogicKnowledgeBase
from loveletter_ai.agents.utils import card_value, remaining_after_play, vulnerable_opponents
from loveletter_ai.belief_expectimax.belief import BayesianBelief


class QLearningAgent:
    """Tabular Q-learning baseline over observation/action feature keys."""

    def __init__(
        self,
        name: str = "QLearningAgent",
        alpha: float = 0.2,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        heuristic_prior_weight: float = 0.35,
        alpha_decay: float = 0.02,
        min_alpha: float = 0.03,
        q_values: dict[tuple, float] | None = None,
        visit_counts: dict[tuple, int] | None = None,
    ) -> None:
        self.name = name
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.heuristic_prior_weight = heuristic_prior_weight
        self.alpha_decay = alpha_decay
        self.min_alpha = min_alpha
        self.prior = ImprovedHeuristicAgent(name=f"{name}Prior")
        self.q_values: dict[tuple, float] = dict(q_values or {})
        self.visit_counts: dict[tuple, int] = dict(visit_counts or {})

    def choose_action(
        self,
        observation: Observation,
        legal_actions: Sequence[Action],
        rng: random.Random,
    ) -> Action:
        if not legal_actions:
            raise ValueError("QLearningAgent received no legal actions.")

        actions = list(legal_actions)
        if self.epsilon > 0 and rng.random() < self.epsilon:
            return rng.choice(actions)

        state_key = observation_key(observation)
        priors = normalized_heuristic_priors(self.prior, observation, actions)
        scored = []
        for index, action in enumerate(actions):
            value = self.action_value(state_key, action, priors[index])
            scored.append((value, index, action))
        best_value = max(value for value, _, _ in scored)
        best_actions = [
            (index, action)
            for value, index, action in scored
            if value == best_value
        ]
        _, best_action = rng.choice(best_actions)
        return best_action

    def update(
        self,
        observation: Observation,
        action: Action,
        reward: float,
        next_observation: Observation | None = None,
        next_legal_actions: Sequence[Action] = (),
    ) -> None:
        state_key = observation_key(observation)
        key = (state_key, action_key(action))
        old_value = self.q_values.get(key, 0.0)

        max_next = 0.0
        if next_observation is not None and next_legal_actions:
            next_key = observation_key(next_observation)
            max_next = max(
                self.q_values.get((next_key, action_key(next_action)), 0.0)
                for next_action in next_legal_actions
            )

        sample = reward + self.gamma * max_next
        step_size = self._effective_alpha(key)
        self.q_values[key] = old_value + step_size * (sample - old_value)

    def action_value(self, state_key: tuple, action: Action, prior: float = 0.0) -> float:
        """Return the policy value used for action selection."""

        q_value = self.q_values.get((state_key, action_key(action)), 0.0)
        return q_value + self.heuristic_prior_weight * prior

    def _effective_alpha(self, key: tuple) -> float:
        visits = self.visit_counts.get(key, 0) + 1
        self.visit_counts[key] = visits
        if self.alpha_decay <= 0:
            return self.alpha
        return max(self.min_alpha, self.alpha / (1.0 + self.alpha_decay * (visits - 1)))

    def save_q_table(self, path: str | Path) -> None:
        records = [
            {
                "state": state_key,
                "action": action,
                "value": value,
                "visits": self.visit_counts.get((state_key, action), 0),
            }
            for (state_key, action), value in self.q_values.items()
        ]
        Path(path).write_text(json.dumps(records, indent=2), encoding="utf-8")

    @classmethod
    def load_q_table(
        cls,
        path: str | Path,
        name: str = "QLearningAgent",
        alpha: float = 0.2,
        gamma: float = 0.95,
        epsilon: float = 0.0,
        heuristic_prior_weight: float = 0.35,
        alpha_decay: float = 0.02,
        min_alpha: float = 0.03,
    ) -> "QLearningAgent":
        records = json.loads(Path(path).read_text(encoding="utf-8"))
        q_values = {
            (_freeze_json(record["state"]), _freeze_json(record["action"])): float(
                record["value"]
            )
            for record in records
        }
        visit_counts = {
            (_freeze_json(record["state"]), _freeze_json(record["action"])): int(
                record.get("visits", 0)
            )
            for record in records
        }
        return cls(
            name=name,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            heuristic_prior_weight=heuristic_prior_weight,
            alpha_decay=alpha_decay,
            min_alpha=min_alpha,
            q_values=q_values,
            visit_counts=visit_counts,
        )


class ApproximateQLearningAgent:
    """Linear approximate Q-learning over compact Love Letter features."""

    def __init__(
        self,
        name: str = "ApproximateQLearningAgent",
        alpha: float = 0.08,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        alpha_decay: float = 0.0005,
        min_alpha: float = 0.01,
        weights: dict[str, float] | None = None,
        initial_heuristic_weight: float = 0.55,
    ) -> None:
        self.name = name
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha_decay = alpha_decay
        self.min_alpha = min_alpha
        self.prior = ImprovedHeuristicAgent(name=f"{name}Prior")
        self.update_count = 0
        self.weights: dict[str, float] = {
            "bias": 0.0,
            "heuristic_score": initial_heuristic_weight,
            "retained_value": 0.08,
            "target_vulnerable": 0.05,
            "discard_princess": -1.0,
            "protect_princess": 0.35,
            "guard_guess_probability": 0.55,
            "guard_guess_value": 0.08,
            "guard_impossible_guess": -0.9,
            "baron_value_edge": 0.3,
            "prince_target_princess_probability": 0.45,
            "self_prince_low_retained": 0.12,
        }
        if weights:
            self.weights.update(weights)

    def choose_action(
        self,
        observation: Observation,
        legal_actions: Sequence[Action],
        rng: random.Random,
    ) -> Action:
        if not legal_actions:
            raise ValueError("ApproximateQLearningAgent received no legal actions.")

        actions = list(legal_actions)
        if self.epsilon > 0 and rng.random() < self.epsilon:
            return rng.choice(actions)

        scored = [
            (self.q_value(observation, action), index, action)
            for index, action in enumerate(actions)
        ]
        best_value = max(value for value, _, _ in scored)
        best = [
            (index, action)
            for value, index, action in scored
            if value == best_value
        ]
        _, action = rng.choice(best)
        return action

    def update(
        self,
        observation: Observation,
        action: Action,
        reward: float,
        next_observation: Observation | None = None,
        next_legal_actions: Sequence[Action] = (),
    ) -> None:
        prediction = self.q_value(observation, action)
        max_next = 0.0
        if next_observation is not None and next_legal_actions:
            max_next = max(
                self.q_value(next_observation, next_action)
                for next_action in next_legal_actions
            )

        target = reward + self.gamma * max_next
        difference = target - prediction
        step_size = self._effective_alpha()
        for feature, value in action_features(self.prior, observation, action).items():
            self.weights[feature] = self.weights.get(feature, 0.0) + step_size * difference * value
            self.weights[feature] = clamp(self.weights[feature], -3.0, 3.0)
        self._regularize_weights()

    def q_value(self, observation: Observation, action: Action) -> float:
        features = action_features(self.prior, observation, action)
        return sum(self.weights.get(feature, 0.0) * value for feature, value in features.items())

    def _effective_alpha(self) -> float:
        self.update_count += 1
        if self.alpha_decay <= 0:
            return self.alpha
        return max(self.min_alpha, self.alpha / (1.0 + self.alpha_decay * (self.update_count - 1)))

    def _regularize_weights(self) -> None:
        """Keep tactically monotone features from flipping sign under noise."""

        non_negative = (
            "heuristic_score",
            "retained_value",
            "protect_princess",
            "guard_guess_probability",
            "guard_guess_value",
            "baron_value_edge",
            "prince_target_princess_probability",
            "self_prince_low_retained",
        )
        non_positive = (
            "discard_princess",
            "guard_impossible_guess",
        )
        for feature in non_negative:
            if feature in self.weights:
                self.weights[feature] = max(0.0, self.weights[feature])
        for feature in non_positive:
            if feature in self.weights:
                self.weights[feature] = min(0.0, self.weights[feature])

    def save_weights(self, path: str | Path) -> None:
        payload = {
            "weights": self.weights,
            "update_count": self.update_count,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load_weights(
        cls,
        path: str | Path,
        name: str = "ApproximateQLearningAgent",
        alpha: float = 0.08,
        gamma: float = 0.95,
        epsilon: float = 0.0,
        alpha_decay: float = 0.0005,
        min_alpha: float = 0.01,
    ) -> "ApproximateQLearningAgent":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        agent = cls(
            name=name,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            alpha_decay=alpha_decay,
            min_alpha=min_alpha,
            weights={key: float(value) for key, value in payload["weights"].items()},
            initial_heuristic_weight=0.0,
        )
        agent.update_count = int(payload.get("update_count", 0))
        return agent


@dataclass(slots=True)
class MCTSNode:
    parent: "MCTSNode | None"
    action: Action | None
    player_to_move: PlayerId | None
    untried_actions: list[Action]
    visits: int = 0
    value: float = 0.0
    children: dict[Action, "MCTSNode"] = field(default_factory=dict)


class MCTSAgent:
    """Perfect-information Monte Carlo tree search over cloned environments.

    This agent intentionally searches the concrete true state and is best used
    as an oracle-style upper-bound baseline. Use ``BeliefMCTSAgent`` for the
    hidden-information version aligned with fair Love Letter play.
    """

    def __init__(
        self,
        name: str = "MCTSAgent",
        simulations: int = 200,
        exploration: float = math.sqrt(2.0),
        rollout_depth: int = 20,
        debug: bool = False,
    ) -> None:
        self.name = name
        self.simulations = simulations
        self.exploration = exploration
        self.rollout_depth = rollout_depth
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
        root_env = env.clone()
        self._advance_to_decision(root_env)
        legal_actions = list(root_env.legal_actions(player_id))
        if not legal_actions:
            raise ValueError("MCTSAgent received no legal actions.")

        root = MCTSNode(
            parent=None,
            action=None,
            player_to_move=player_id,
            untried_actions=list(legal_actions),
        )

        for _ in range(max(1, self.simulations)):
            sim_env = root_env.clone()
            node = root
            path = [node]

            while not node.untried_actions and node.children:
                action, node = self._select_child(node, rng)
                sim_env.step(action)
                self._advance_to_decision(sim_env)
                path.append(node)

            if self._is_terminal(sim_env):
                result = self._terminal_value(sim_env, player_id)
            else:
                self._ensure_untried_actions(node, sim_env)
                if node.untried_actions:
                    action = rng.choice(node.untried_actions)
                    node.untried_actions.remove(action)
                    sim_env.step(action)
                    self._advance_to_decision(sim_env)
                    child = MCTSNode(
                        parent=node,
                        action=action,
                        player_to_move=self._current_player(sim_env),
                        untried_actions=self._legal_actions_for_current(sim_env),
                    )
                    node.children[action] = child
                    node = child
                    path.append(node)
                result = self._rollout(sim_env, player_id, rng)

            for visited in path:
                visited.visits += 1
                visited.value += result

        if not root.children:
            observation = env.observe(player_id)
            return self.fallback.choose_action(observation, legal_actions, rng)

        best_children = sorted(
            root.children.items(),
            key=lambda item: (
                item[1].visits,
                item[1].value / item[1].visits if item[1].visits else float("-inf"),
            ),
            reverse=True,
        )
        best_visits = best_children[0][1].visits
        best_mean = (
            best_children[0][1].value / best_children[0][1].visits
            if best_children[0][1].visits
            else float("-inf")
        )
        tied = [
            action
            for action, child in best_children
            if child.visits == best_visits
            and (child.value / child.visits if child.visits else float("-inf"))
            == best_mean
        ]
        if self.debug:
            for action, child in best_children:
                mean = child.value / child.visits if child.visits else 0.0
                print(
                    f"{self.name} eval {action.label()}: "
                    f"visits={child.visits} value={mean:.2f}"
                )
        return rng.choice(tied)

    def _select_child(
        self,
        node: MCTSNode,
        rng: random.Random,
    ) -> tuple[Action, MCTSNode]:
        log_parent = math.log(max(1, node.visits))
        scored = []
        for index, (action, child) in enumerate(node.children.items()):
            if child.visits == 0:
                score = float("inf")
            else:
                exploit = child.value / child.visits
                explore = self.exploration * math.sqrt(log_parent / child.visits)
                score = exploit + explore
            scored.append((score, index, action, child))

        best_score = max(score for score, _, _, _ in scored)
        best = [
            (action, child)
            for score, _, action, child in scored
            if score == best_score
        ]
        return rng.choice(best)

    def _rollout(self, env, root_player: PlayerId, rng: random.Random) -> float:
        depth = 0
        while not self._is_terminal(env) and depth < self.rollout_depth:
            self._advance_to_decision(env)
            if self._is_terminal(env):
                break
            actor = self._current_player(env)
            if actor is None:
                break
            legal_actions = env.legal_actions(actor)
            if not legal_actions:
                break
            if actor == root_player:
                observation = env.observe(actor)
                action = self.fallback.choose_action(observation, legal_actions, rng)
            else:
                action = rng.choice(list(legal_actions))
            env.step(action)
            depth += 1

        if self._is_terminal(env):
            return self._terminal_value(env, root_player)
        return max(-1.0, min(1.0, evaluate_position(env, root_player) / 100.0))

    def _advance_to_decision(self, env) -> None:
        state = env.state
        while state is not None and state.phase == Phase.DRAW:
            state = env.begin_action_phase()

    def _ensure_untried_actions(self, node: MCTSNode, env) -> None:
        if not node.untried_actions and not node.children:
            node.untried_actions = self._legal_actions_for_current(env)

    def _legal_actions_for_current(self, env) -> list[Action]:
        actor = self._current_player(env)
        if actor is None:
            return []
        return list(env.legal_actions(actor))

    def _current_player(self, env) -> PlayerId | None:
        if env.state is None:
            return None
        return env.state.current_player

    def _is_terminal(self, env) -> bool:
        return env.state is not None and env.state.phase == Phase.ROUND_OVER

    def _terminal_value(self, env, root_player: PlayerId) -> float:
        state = env.state
        if state is None or state.winner is None:
            return 0.0
        return 1.0 if state.winner == root_player else -1.0


class BeliefMCTSAgent(MCTSAgent):
    """Information-set MCTS with root hidden-state sampling.

    Each simulation samples opponent hands, the hidden card, and deck order from
    the current observation-derived belief before evaluating a root action. The
    true environment is used only as a public transition skeleton; hidden fields
    are overwritten by the sampled determinization before rollouts begin.
    """

    def __init__(
        self,
        name: str = "BeliefMCTSAgent",
        simulations: int = 200,
        exploration: float = math.sqrt(2.0),
        rollout_depth: int = 20,
        debug: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            simulations=simulations,
            exploration=exploration,
            rollout_depth=rollout_depth,
            debug=debug,
        )
        self.kb = LogicKnowledgeBase()

    def choose_action_from_env(
        self,
        env,
        player_id: PlayerId,
        rng: random.Random,
    ) -> Action:
        root_env = env.clone()
        self._advance_to_decision(root_env)
        observation = root_env.observe(player_id)
        legal_actions = list(root_env.legal_actions(player_id))
        if not legal_actions:
            raise ValueError("BeliefMCTSAgent received no legal actions.")

        self.kb.update(observation)
        possible_cards = {
            opponent: self.kb.possible_cards(opponent)
            for opponent in self.kb.possible
        }
        belief = BayesianBelief.from_observation(observation, possible_cards)

        stats = {
            action: MCTSNode(
                parent=None,
                action=action,
                player_to_move=player_id,
                untried_actions=[],
            )
            for action in legal_actions
        }

        for _ in range(max(1, self.simulations)):
            action = self._select_root_action(stats, rng)
            sim_env = root_env.clone()
            sample = belief.sample_hidden_state(rng)
            belief.apply_sample_to_env(sim_env, sample)

            if action not in sim_env.legal_actions(player_id):
                # The sampled hand must preserve the viewer's own legal actions.
                # If a future extension changes that assumption, treat this
                # simulation as neutral instead of leaking true-state details.
                result = 0.0
            else:
                sim_env.step(action)
                result = self._rollout(sim_env, player_id, rng)

            node = stats[action]
            node.visits += 1
            node.value += result

        best = sorted(
            stats.values(),
            key=lambda node: (
                node.visits,
                node.value / node.visits if node.visits else float("-inf"),
            ),
            reverse=True,
        )
        best_visits = best[0].visits
        best_mean = best[0].value / best[0].visits if best[0].visits else float("-inf")
        tied = [
            node.action
            for node in best
            if node.visits == best_visits
            and (node.value / node.visits if node.visits else float("-inf"))
            == best_mean
            and node.action is not None
        ]

        if self.debug:
            for node in best:
                if node.action is None:
                    continue
                mean = node.value / node.visits if node.visits else 0.0
                print(
                    f"{self.name} eval {node.action.label()}: "
                    f"visits={node.visits} value={mean:.2f}"
                )
        return rng.choice(tied)

    def observe_action(self, action: Action) -> None:
        self.kb.note_action(action)

    def _select_root_action(
        self,
        stats: dict[Action, MCTSNode],
        rng: random.Random,
    ) -> Action:
        unvisited = [node.action for node in stats.values() if node.visits == 0]
        unvisited = [action for action in unvisited if action is not None]
        if unvisited:
            return rng.choice(unvisited)

        total_visits = sum(node.visits for node in stats.values())
        log_total = math.log(max(1, total_visits))
        scored = []
        for index, node in enumerate(stats.values()):
            if node.action is None:
                continue
            exploit = node.value / node.visits
            explore = self.exploration * math.sqrt(log_total / node.visits)
            scored.append((exploit + explore, index, node.action))

        best_score = max(score for score, _, _ in scored)
        best = [action for score, _, action in scored if score == best_score]
        return rng.choice(best)


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


def action_key(action: Action) -> tuple:
    return (
        action.card.value,
        int(action.target) if action.target is not None else None,
        action.guess.value if action.guess is not None else None,
    )


def evaluate_position(env, root_player: PlayerId) -> float:
    state = env.state
    if state is None:
        raise RuntimeError("Cannot evaluate an environment before reset.")

    if state.phase == Phase.ROUND_OVER:
        if state.winner == root_player:
            return 100.0
        if state.winner is None:
            return 0.0
        return -100.0

    root_state = state.player_states[root_player]
    if root_state.eliminated:
        return -90.0

    score = _position_value(root_state)
    for player in state.players:
        if player.id == root_player:
            continue
        opponent = state.player_states[player.id]
        if opponent.eliminated:
            score += 30.0
        else:
            score -= 0.5 * _position_value(opponent)
    return score


def _position_value(player_state) -> float:
    hand_value = max((card_value(card) for card in player_state.hand), default=0)
    score = hand_value * 10.0 + player_state.discard_total
    if player_state.protected:
        score += 8.0
    if CardName.PRINCESS in player_state.hand:
        score += 5.0
    return score


def _freeze_json(value):
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def normalized_heuristic_priors(
    prior: ImprovedHeuristicAgent,
    observation: Observation,
    actions: Sequence[Action],
) -> list[float]:
    scores = [prior.score_action(observation, action) for action in actions]
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if high == low:
        return [0.0 for _ in scores]
    return [(score - low) / (high - low) for score in scores]


def action_features(
    prior: ImprovedHeuristicAgent,
    observation: Observation,
    action: Action,
) -> dict[str, float]:
    """Return normalized action features for approximate Q-learning."""

    hand = list(observation.own_hand)
    remaining = remaining_after_play(hand, action.card)
    retained_value = max((card_value(card) for card in remaining), default=0)
    counts = unseen_card_counts(observation)
    unseen_total = sum(counts.values())
    expected_value = expected_card_value(counts)
    vulnerable = vulnerable_opponents(observation)
    raw_heuristic = prior.score_action(observation, action)

    features = {
        "bias": 1.0,
        "heuristic_score": clamp(raw_heuristic / 300.0, -1.0, 1.0),
        "played_value": card_value(action.card) / 8.0,
        "retained_value": retained_value / 8.0,
        "target_vulnerable": 1.0 if action.target in vulnerable else 0.0,
        "hand_has_princess": 1.0 if CardName.PRINCESS in hand else 0.0,
        "discard_princess": 1.0 if action.card == CardName.PRINCESS else 0.0,
        "protect_princess": 1.0
        if CardName.PRINCESS in hand and action.card == CardName.HANDMAID
        else 0.0,
        "play_handmaid": 1.0 if action.card == CardName.HANDMAID else 0.0,
        "play_guard": 1.0 if action.card == CardName.GUARD else 0.0,
        "play_baron": 1.0 if action.card == CardName.BARON else 0.0,
        "play_prince": 1.0 if action.card == CardName.PRINCE else 0.0,
        "play_king": 1.0 if action.card == CardName.KING else 0.0,
    }

    if action.card == CardName.GUARD and action.guess is not None:
        guess_count = counts.get(action.guess, 0)
        features["guard_guess_probability"] = (
            guess_count / unseen_total if unseen_total else 0.0
        )
        features["guard_guess_value"] = card_value(action.guess) / 8.0
        features["guard_impossible_guess"] = 1.0 if guess_count <= 0 else 0.0
    else:
        features["guard_guess_probability"] = 0.0
        features["guard_guess_value"] = 0.0
        features["guard_impossible_guess"] = 0.0

    if action.card == CardName.BARON:
        features["baron_value_edge"] = clamp(
            (retained_value - expected_value) / 8.0,
            -1.0,
            1.0,
        )
    else:
        features["baron_value_edge"] = 0.0

    if action.card == CardName.PRINCE and action.target != observation.viewer:
        princess_count = counts.get(CardName.PRINCESS, 0)
        features["prince_target_princess_probability"] = (
            princess_count / unseen_total if unseen_total else 0.0
        )
    else:
        features["prince_target_princess_probability"] = 0.0

    if action.card == CardName.PRINCE and action.target == observation.viewer:
        features["self_prince_low_retained"] = 1.0 if retained_value <= 2 else 0.0
    else:
        features["self_prince_low_retained"] = 0.0

    return features


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
