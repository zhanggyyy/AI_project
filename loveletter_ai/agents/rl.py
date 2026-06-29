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

from loveletter_ai.agents.heuristics import ImprovedHeuristicAgent
from loveletter_ai.agents.logic import LogicKnowledgeBase
from loveletter_ai.agents.utils import card_value
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
        q_values: dict[tuple, float] | None = None,
    ) -> None:
        self.name = name
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.heuristic_prior_weight = heuristic_prior_weight
        self.prior = ImprovedHeuristicAgent(name=f"{name}Prior")
        self.q_values: dict[tuple, float] = dict(q_values or {})

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
            q_value = self.q_values.get((state_key, action_key(action)), 0.0)
            value = q_value + self.heuristic_prior_weight * priors[index]
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
        self.q_values[key] = old_value + self.alpha * (sample - old_value)

    def save_q_table(self, path: str | Path) -> None:
        records = [
            {"state": state_key, "action": action, "value": value}
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
    ) -> "QLearningAgent":
        records = json.loads(Path(path).read_text(encoding="utf-8"))
        q_values = {
            (_freeze_json(record["state"]), _freeze_json(record["action"])): float(
                record["value"]
            )
            for record in records
        }
        return cls(
            name=name,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            heuristic_prior_weight=heuristic_prior_weight,
            q_values=q_values,
        )


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
