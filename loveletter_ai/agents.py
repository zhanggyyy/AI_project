"""Agent interfaces and baseline reasoning agents."""

from __future__ import annotations

import random
import re
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

from loveletter_ai.actions import Action
from loveletter_ai.cards import CARD_SPECS, CardName
from loveletter_ai.players import PlayerId
from loveletter_ai.state import Observation, Phase, PublicPlayerView


class Agent(Protocol):
    """Minimal policy interface used by the environment runner."""

    name: str

    def choose_action(
        self,
        observation: Observation,
        legal_actions: Sequence[Action],
        rng: random.Random,
    ) -> Action:
        """Choose one action from an environment-provided legal action list."""


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
    """Simple rule-based agent with no persistent hidden-state model."""

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


@dataclass(slots=True)
class LogicKnowledgeBase:
    """Symbolic card-set constraints for hidden opponent hands.

    The KB stores possible current hand cards per opponent. It intentionally does
    not assign probabilities. Bayesian agents can later reuse the same
    observation parsing and constraint events as likelihood inputs.
    """

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

    def note_action(self, action: Action) -> None:
        if action.card == CardName.GUARD and action.guess is not None:
            self.pending_guard_guesses.append((action.target, action.guess))
            self._deduce(f"Observed Guard guess: {action.guess.value}.")

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
    """Symbolic reasoning agent backed by a persistent knowledge base."""

    def __init__(self, name: str = "LogicAgent", debug: bool = False) -> None:
        self.name = name
        self.debug = debug
        self.kb = LogicKnowledgeBase()
        self.fallback = NaiveHeuristicAgent(name=f"{name}Fallback")

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
        """Allow runners to feed public action metadata into the KB."""

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


class ExpectimaxAgent:
    """Depth-limited expectimax over a cloned deterministic environment.

    This first version is a determinized research baseline: it searches the
    current concrete environment state. It does not yet sample hidden states from
    a belief model. The normal `choose_action` method falls back to the naive
    heuristic because observations alone are not enough to simulate transitions.
    """

    def __init__(
        self,
        name: str = "ExpectimaxAgent",
        depth: int = 3,
        debug: bool = False,
    ) -> None:
        self.name = name
        self.depth = depth
        self.debug = debug
        self.fallback = NaiveHeuristicAgent(name=f"{name}Fallback")

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
        if state.phase.value == "round_over" or depth <= 0:
            return self._evaluate(env, root_player)

        if state.phase.value == "draw":
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

        if state.phase.value == "round_over":
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


class QLearningAgent:
    """Tabular Q-learning baseline over observation/action feature keys."""

    def __init__(
        self,
        name: str = "QLearningAgent",
        alpha: float = 0.2,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        q_values: dict[tuple, float] | None = None,
    ) -> None:
        self.name = name
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
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
        scored = [
            (self.q_values.get((state_key, action_key(action)), 0.0), index, action)
            for index, action in enumerate(actions)
        ]
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
    ) -> "QLearningAgent":
        records = json.loads(Path(path).read_text(encoding="utf-8"))
        q_values = {
            (_freeze_json(record["state"]), _freeze_json(record["action"])): float(
                record["value"]
            )
            for record in records
        }
        return cls(name=name, alpha=alpha, gamma=gamma, epsilon=epsilon, q_values=q_values)


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
    """Monte Carlo Tree Search agent over cloned deterministic environments."""

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
        self.fallback = NaiveHeuristicAgent(name=f"{name}Fallback")

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
            and (
                child.value / child.visits if child.visits else float("-inf")
            )
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


def remaining_after_play(hand: list[CardName], played: CardName) -> list[CardName]:
    remaining = list(hand)
    if played in remaining:
        remaining.remove(played)
    return remaining


def card_value(card: CardName) -> int:
    return CARD_SPECS[card].value


def card_from_label(label: str) -> CardName | None:
    normalized = label.strip().rstrip(".")
    for card in CardName:
        if card.value == normalized:
            return card
    return None
