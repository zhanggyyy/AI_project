"""Run a full match with RandomAgent on both seats."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loveletter_ai.agents import RandomAgent
from loveletter_ai.environment import LoveLetterEnv
from loveletter_ai.renderers import TerminalRenderer
from loveletter_ai.state import Phase


def main() -> None:
    run_match(RandomAgent("Random-Alice"), RandomAgent("Random-Bob"))


def run_match(agent_a, agent_b) -> None:
    env = LoveLetterEnv(["Alice", "Bob"], seed=188)
    agents = {env.players[0].id: agent_a, env.players[1].id: agent_b}
    renderer = TerminalRenderer()
    state = env.reset()
    print(renderer.render_state(state, reveal=True))

    while state.phase != Phase.ROUND_OVER:
        state = env.begin_action_phase()
        if state.phase == Phase.ROUND_OVER:
            break
        actor = state.current_player
        action = choose_agent_action(agents[actor], env, actor)
        print(f"\n{agents[actor].name}: {action.label()}")
        notify_agents(agents.values(), action)
        state = env.step(action)
        print(renderer.render_state(state, reveal=True))

    print_winner(state)


def notify_agents(agents, action) -> None:
    for agent in agents:
        observe_action = getattr(agent, "observe_action", None)
        if observe_action is not None:
            observe_action(action)


def choose_agent_action(agent, env, actor):
    search_action = getattr(agent, "choose_action_from_env", None)
    if search_action is not None:
        return search_action(env, actor, env.rng)

    observation = env.observe(actor)
    return agent.choose_action(observation, env.legal_actions(actor), env.rng)


def print_winner(state) -> None:
    if state.winner is None:
        print("\nWinner: none")
        return
    winner = next(player for player in state.players if player.id == state.winner)
    print(f"\nWinner: P{int(winner.id)} {winner.name}")


if __name__ == "__main__":
    main()
