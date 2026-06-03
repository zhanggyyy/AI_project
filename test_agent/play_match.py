"""Interactive terminal match runner for AI-vs-AI or human-vs-AI games."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loveletter_ai.agents import (
    ExpectimaxAgent,
    LogicAgent,
    MCTSAgent,
    NaiveHeuristicAgent,
    QLearningAgent,
    RandomAgent,
)
from loveletter_ai.environment import LoveLetterEnv
from loveletter_ai.renderers import TerminalRenderer
from loveletter_ai.state import Phase
from loveletter_ai.training import train_q_learning_self_play


class HumanTerminalAgent:
    def __init__(self, name: str = "Human") -> None:
        self.name = name


AGENT_OPTIONS = {
    "1": "random",
    "2": "naive",
    "3": "logic",
    "4": "expectimax",
    "5": "qlearning",
    "6": "mcts",
    "7": "human",
}


def main() -> None:
    print("Love Letter match runner")
    print_agent_menu()

    alice_kind = prompt_agent_kind("Alice")
    bob_kind = prompt_agent_kind("Bob")
    seed = prompt_int("Seed", default=188)

    env = LoveLetterEnv(["Alice", "Bob"], seed=seed)
    agents = {
        env.players[0].id: build_agent(alice_kind, "Alice"),
        env.players[1].id: build_agent(bob_kind, "Bob"),
    }
    run_interactive_match(env, agents)


def print_agent_menu() -> None:
    print("\nChoose an agent:")
    for key, name in AGENT_OPTIONS.items():
        print(f"  {key}. {name}")


def prompt_agent_kind(player_name: str) -> str:
    while True:
        choice = input(f"{player_name} agent number: ").strip().lower()
        if choice in AGENT_OPTIONS:
            return AGENT_OPTIONS[choice]
        if choice in AGENT_OPTIONS.values():
            return choice
        print("Please choose a listed number or name.")


def build_agent(kind: str, seat_name: str):
    prefix = kind.capitalize()
    if kind == "random":
        return RandomAgent(f"Random-{seat_name}")
    if kind == "naive":
        return NaiveHeuristicAgent(f"Naive-{seat_name}")
    if kind == "logic":
        return LogicAgent(f"Logic-{seat_name}", debug=False)
    if kind == "expectimax":
        depth = prompt_int(f"{seat_name} Expectimax depth", default=3)
        return ExpectimaxAgent(f"Expectimax-{seat_name}", depth=depth, debug=False)
    if kind == "qlearning":
        episodes = prompt_int(f"{seat_name} Q-learning training episodes", default=200)
        if episodes > 0:
            agent = train_q_learning_self_play(
                episodes=episodes,
                seed=188 + len(seat_name),
                epsilon=0.25,
            )
            agent.name = f"Q-{seat_name}"
            return agent
        return QLearningAgent(f"Q-{seat_name}", epsilon=0.0)
    if kind == "mcts":
        simulations = prompt_int(f"{seat_name} MCTS simulations", default=120)
        return MCTSAgent(
            f"MCTS-{seat_name}",
            simulations=simulations,
            rollout_depth=20,
            debug=False,
        )
    if kind == "human":
        return HumanTerminalAgent(f"Human-{seat_name}")
    raise ValueError(f"Unknown agent kind: {prefix}")


def prompt_int(label: str, default: int) -> int:
    raw = input(f"{label} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"Invalid integer, using {default}.")
        return default


def run_interactive_match(env, agents) -> None:
    renderer = TerminalRenderer()
    state = env.reset()
    print("\n" + renderer.render_state(state, reveal=True))

    while state.phase != Phase.ROUND_OVER:
        state = env.begin_action_phase()
        if state.phase == Phase.ROUND_OVER:
            break

        actor = state.current_player
        agent = agents[actor]
        action = choose_action(agent, env, actor, renderer)
        print(f"\n{agent.name}: {action.label()}")
        notify_agents(agents.values(), action)
        state = env.step(action)

        reveal = not any(isinstance(agent, HumanTerminalAgent) for agent in agents.values())
        print("\n" + renderer.render_state(state, viewer=actor, reveal=reveal))

    print_winner(state)


def choose_action(agent, env, actor, renderer):
    legal_actions = env.legal_actions(actor)
    if isinstance(agent, HumanTerminalAgent):
        print("\n" + renderer.render_state(env.state, viewer=actor, reveal=False))
        print(renderer.render_actions(legal_actions))
        while True:
            raw = input("Choose action index: ").strip()
            try:
                index = int(raw)
            except ValueError:
                print("Please enter a number.")
                continue
            if 0 <= index < len(legal_actions):
                return legal_actions[index]
            print("Action index out of range.")

    search_action = getattr(agent, "choose_action_from_env", None)
    if search_action is not None:
        return search_action(env, actor, env.rng)

    observation = env.observe(actor)
    return agent.choose_action(observation, legal_actions, env.rng)


def notify_agents(agents, action) -> None:
    for agent in agents:
        observe_action = getattr(agent, "observe_action", None)
        if observe_action is not None:
            observe_action(action)


def print_winner(state) -> None:
    if state.winner is None:
        print("\nWinner: none")
        return
    winner = next(player for player in state.players if player.id == state.winner)
    print(f"\nWinner: P{int(winner.id)} {winner.name}")


if __name__ == "__main__":
    main()
