"""Small training helpers for reinforcement-learning agents."""

from __future__ import annotations

import random

from loveletter_ai.actions import Action
from loveletter_ai.agents import QLearningAgent
from loveletter_ai.environment import LoveLetterEnv
from loveletter_ai.players import PlayerId
from loveletter_ai.state import Observation, Phase


def train_q_learning_self_play(
    episodes: int,
    seed: int | None = None,
    epsilon: float = 0.2,
    alpha: float = 0.2,
    gamma: float = 0.95,
) -> QLearningAgent:
    """Train one shared tabular Q-learning policy by two-player self-play."""

    trainer_rng = random.Random(seed)
    agent = QLearningAgent(
        name="QLearningAgent",
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
    )

    for _ in range(episodes):
        episode_seed = trainer_rng.randrange(1_000_000_000)
        env = LoveLetterEnv(["Alice", "Bob"], seed=episode_seed)
        state = env.reset()
        last_transition: dict[PlayerId, tuple[Observation, Action]] = {}

        while state.phase != Phase.ROUND_OVER:
            state = env.begin_action_phase()
            if state.phase == Phase.ROUND_OVER:
                break

            actor = state.current_player
            observation = env.observe(actor)
            legal_actions = env.legal_actions(actor)
            action = agent.choose_action(observation, legal_actions, env.rng)

            if actor in last_transition:
                prev_observation, prev_action = last_transition[actor]
                agent.update(prev_observation, prev_action, 0.0, observation, legal_actions)

            state = env.step(action)
            last_transition[actor] = (observation, action)

        for player_id, (observation, action) in last_transition.items():
            reward = 1.0 if state.winner == player_id else -1.0
            agent.update(observation, action, reward, None, ())

    agent.epsilon = 0.0
    return agent
