"""Training helpers for reinforcement-learning agents."""

from __future__ import annotations

import random
from collections.abc import Iterable

from loveletter_ai.actions import Action
from loveletter_ai.agents import (
    ApproximateQLearningAgent,
    ImprovedHeuristicAgent,
    NaiveHeuristicAgent,
    QLearningAgent,
    RandomAgent,
)
from loveletter_ai.cards import CardName
from loveletter_ai.environment import LoveLetterEnv
from loveletter_ai.players import PlayerId
from loveletter_ai.state import GameState, Observation, Phase


OpponentKind = str


def train_q_learning_self_play(
    episodes: int,
    seed: int | None = None,
    epsilon: float = 0.2,
    alpha: float = 0.2,
    gamma: float = 0.95,
) -> QLearningAgent:
    """Train a tabular Q-learning policy by two-player self-play.

    This wrapper preserves the original public API. The implementation now
    uses decaying exploration, decaying per-state learning rates, and reward
    shaping.
    """

    return train_q_learning_mixed(
        episodes=episodes,
        seed=seed,
        agent_kind="tabular",
        epsilon_start=epsilon,
        epsilon_end=0.02,
        alpha=alpha,
        gamma=gamma,
        opponent_pool=("self",),
    )


def train_q_learning_mixed(
    episodes: int,
    seed: int | None = None,
    agent_kind: str = "tabular",
    epsilon_start: float = 0.4,
    epsilon_end: float = 0.02,
    alpha: float | None = None,
    gamma: float = 0.95,
    opponent_pool: Iterable[OpponentKind] = ("random", "naive", "improved", "self"),
    heuristic_prior_weight: float = 0.35,
    shaping: bool = True,
) -> QLearningAgent | ApproximateQLearningAgent:
    """Train one learner against a mixed opponent curriculum.

    The mixed pool reduces overfitting to a single non-stationary self-play
    opponent. The returned policy has exploration disabled for evaluation.
    """

    if episodes < 0:
        raise ValueError("episodes must be non-negative.")

    trainer_rng = random.Random(seed)
    opponents = tuple(opponent_pool)
    if not opponents:
        raise ValueError("opponent_pool must contain at least one opponent kind.")

    if agent_kind == "tabular":
        agent: QLearningAgent | ApproximateQLearningAgent = QLearningAgent(
            name="QLearningAgent",
            alpha=0.16 if alpha is None else alpha,
            gamma=gamma,
            epsilon=epsilon_start,
            heuristic_prior_weight=heuristic_prior_weight,
            alpha_decay=0.04,
            min_alpha=0.025,
        )
    elif agent_kind == "approximate":
        agent = ApproximateQLearningAgent(
            name="ApproximateQLearningAgent",
            alpha=0.07 if alpha is None else alpha,
            gamma=gamma,
            epsilon=epsilon_start,
            alpha_decay=0.0008,
            min_alpha=0.008,
            initial_heuristic_weight=0.55,
        )
    else:
        raise ValueError("agent_kind must be 'tabular' or 'approximate'.")

    for episode in range(episodes):
        progress = episode / max(1, episodes - 1)
        agent.epsilon = epsilon_start + progress * (epsilon_end - epsilon_start)
        opponent_kind = trainer_rng.choice(opponents)
        _run_training_episode(
            agent,
            opponent_kind,
            seed=trainer_rng.randrange(1_000_000_000),
            trainer_rng=trainer_rng,
            shaping=shaping,
        )

    agent.epsilon = 0.0
    return agent


def _run_training_episode(
    agent: QLearningAgent | ApproximateQLearningAgent,
    opponent_kind: OpponentKind,
    seed: int,
    trainer_rng: random.Random,
    shaping: bool,
) -> None:
    env = LoveLetterEnv(["Alice", "Bob"], seed=seed)
    state = env.reset()
    learner_seat = trainer_rng.choice([env.players[0].id, env.players[1].id])
    if opponent_kind == "self":
        learner_players = {env.players[0].id, env.players[1].id}
        opponent = None
    else:
        learner_players = {learner_seat}
        opponent = _make_opponent(opponent_kind)

    last_transition: dict[PlayerId, tuple[Observation, Action]] = {}
    pending_reward = {player_id: 0.0 for player_id in learner_players}

    while state.phase != Phase.ROUND_OVER:
        state = env.begin_action_phase()
        if state.phase == Phase.ROUND_OVER:
            break

        actor = state.current_player
        if actor is None:
            raise RuntimeError("No current player is set.")

        observation = env.observe(actor)
        legal_actions = env.legal_actions(actor)
        if actor in learner_players:
            if actor in last_transition:
                prev_observation, prev_action = last_transition[actor]
                agent.update(
                    prev_observation,
                    prev_action,
                    pending_reward[actor],
                    observation,
                    legal_actions,
                )
                pending_reward[actor] = 0.0
            action = agent.choose_action(observation, legal_actions, env.rng)
        else:
            if opponent is None:
                raise RuntimeError("Non-learner actor has no opponent policy.")
            action = opponent.choose_action(observation, legal_actions, env.rng)

        before = _snapshot_player_states(state)
        state = env.step(action)
        if shaping:
            shaped = _shaped_rewards(before, state, action)
            for player_id in learner_players:
                pending_reward[player_id] += shaped.get(player_id, 0.0)

        if actor in learner_players:
            last_transition[actor] = (observation, action)

    for player_id, (observation, action) in last_transition.items():
        terminal = 1.0 if state.winner == player_id else -1.0
        agent.update(observation, action, terminal + pending_reward[player_id], None, ())


def _make_opponent(kind: OpponentKind):
    if kind == "random":
        return RandomAgent("TrainingRandom")
    if kind == "naive":
        return NaiveHeuristicAgent("TrainingNaive")
    if kind == "improved":
        return ImprovedHeuristicAgent("TrainingImproved")
    raise ValueError(f"Unknown opponent kind: {kind}")


def _snapshot_player_states(state: GameState) -> dict[PlayerId, dict[str, object]]:
    return {
        player.id: {
            "hand": tuple(state.player_states[player.id].hand),
            "discard": tuple(state.player_states[player.id].discard_pile),
            "eliminated": state.player_states[player.id].eliminated,
        }
        for player in state.players
    }


def _shaped_rewards(
    before: dict[PlayerId, dict[str, object]],
    after: GameState,
    action: Action,
) -> dict[PlayerId, float]:
    """Return small dense rewards for events with clear tactical meaning."""

    rewards = {player_id: 0.0 for player_id in before}
    actor = action.actor

    if action.card == CardName.PRINCESS:
        rewards[actor] -= 0.8

    if action.card == CardName.HANDMAID and CardName.PRINCESS in before[actor]["hand"]:
        rewards[actor] += 0.08

    if action.card == CardName.PRIEST and action.target is not None:
        rewards[actor] += 0.04

    target_eliminated = False
    if action.target is not None:
        target_before = bool(before[action.target]["eliminated"])
        target_after = after.player_states[action.target].eliminated
        target_eliminated = not target_before and target_after

    for player_id, snapshot in before.items():
        was_eliminated = bool(snapshot["eliminated"])
        is_eliminated = after.player_states[player_id].eliminated
        if was_eliminated or not is_eliminated:
            continue
        if player_id == actor:
            rewards[player_id] -= 0.4
        else:
            rewards[actor] += 0.3
            rewards[player_id] -= 0.3

    if action.card == CardName.GUARD and target_eliminated:
        rewards[actor] += 0.18
    if action.card == CardName.BARON and target_eliminated:
        rewards[actor] += 0.12
    if action.card == CardName.PRINCE and target_eliminated:
        rewards[actor] += 0.12

    return rewards
