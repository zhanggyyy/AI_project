"""Shared match-running and evaluation helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from loveletter_ai.players import PlayerId
from loveletter_ai.environment import LoveLetterEnv
from loveletter_ai.state import Phase


AgentFactory = Callable[[], object]


@dataclass(frozen=True, slots=True)
class RoundResult:
    """Summary of one completed round."""

    winner: PlayerId | None
    turns: int
    seed: int


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Aggregate result for a two-agent evaluation."""

    agent_a: str
    agent_b: str
    games: int
    wins_a: int
    wins_b: int
    draws: int

    @property
    def win_rate_a(self) -> float:
        return self.wins_a / self.games if self.games else 0.0

    @property
    def win_rate_b(self) -> float:
        return self.wins_b / self.games if self.games else 0.0


def run_round(
    agent_a: object,
    agent_b: object,
    seed: int,
    player_names: tuple[str, str] = ("Alice", "Bob"),
) -> RoundResult:
    """Run one two-player round and return its result."""

    env = LoveLetterEnv(player_names, seed=seed)
    agents = {env.players[0].id: agent_a, env.players[1].id: agent_b}
    state = env.reset()

    while state.phase != Phase.ROUND_OVER:
        state = env.begin_action_phase()
        if state.phase == Phase.ROUND_OVER:
            break
        actor = state.current_player
        if actor is None:
            raise RuntimeError("No current player is set.")
        action = choose_agent_action(agents[actor], env, actor)
        notify_agents(agents.values(), action)
        state = env.step(action)

    return RoundResult(winner=state.winner, turns=state.turn_index, seed=seed)


def evaluate_pair(
    agent_a_factory: AgentFactory,
    agent_b_factory: AgentFactory,
    games: int = 100,
    seed: int = 0,
) -> EvaluationResult:
    """Evaluate fresh agent instances across deterministic seeds."""

    if games < 1:
        raise ValueError("games must be at least 1.")

    wins_a = 0
    wins_b = 0
    draws = 0
    agent_a_name = ""
    agent_b_name = ""

    for offset in range(games):
        agent_a = agent_a_factory()
        agent_b = agent_b_factory()
        agent_a_name = getattr(agent_a, "name", agent_a.__class__.__name__)
        agent_b_name = getattr(agent_b, "name", agent_b.__class__.__name__)
        result = run_round(agent_a, agent_b, seed + offset)

        if result.winner == PlayerId(0):
            wins_a += 1
        elif result.winner == PlayerId(1):
            wins_b += 1
        else:
            draws += 1

    return EvaluationResult(
        agent_a=agent_a_name,
        agent_b=agent_b_name,
        games=games,
        wins_a=wins_a,
        wins_b=wins_b,
        draws=draws,
    )


def evaluate_pair_symmetric(
    agent_a_factory: AgentFactory,
    agent_b_factory: AgentFactory,
    games_per_seat: int = 50,
    seed: int = 0,
) -> EvaluationResult:
    """Evaluate both seat assignments to reduce first-seat seed bias."""

    if games_per_seat < 1:
        raise ValueError("games_per_seat must be at least 1.")

    wins_a = 0
    wins_b = 0
    draws = 0
    agent_a_name = ""
    agent_b_name = ""

    for offset in range(games_per_seat):
        agent_a = agent_a_factory()
        agent_b = agent_b_factory()
        agent_a_name = getattr(agent_a, "name", agent_a.__class__.__name__)
        agent_b_name = getattr(agent_b, "name", agent_b.__class__.__name__)
        result = run_round(agent_a, agent_b, seed + offset)
        if result.winner == PlayerId(0):
            wins_a += 1
        elif result.winner == PlayerId(1):
            wins_b += 1
        else:
            draws += 1

        agent_a = agent_a_factory()
        agent_b = agent_b_factory()
        result = run_round(agent_b, agent_a, seed + games_per_seat + offset)
        if result.winner == PlayerId(1):
            wins_a += 1
        elif result.winner == PlayerId(0):
            wins_b += 1
        else:
            draws += 1

    return EvaluationResult(
        agent_a=agent_a_name,
        agent_b=agent_b_name,
        games=games_per_seat * 2,
        wins_a=wins_a,
        wins_b=wins_b,
        draws=draws,
    )


def choose_agent_action(agent: object, env: LoveLetterEnv, actor: PlayerId):
    """Choose an action using the search extension when available."""

    search_action = getattr(agent, "choose_action_from_env", None)
    if search_action is not None:
        return search_action(env, actor, env.rng)

    observation = env.observe(actor)
    legal_actions = env.legal_actions(actor)
    return agent.choose_action(observation, legal_actions, env.rng)


def notify_agents(agents: Iterable[object], action) -> None:
    for agent in agents:
        observe_action = getattr(agent, "observe_action", None)
        if observe_action is not None:
            observe_action(action)
