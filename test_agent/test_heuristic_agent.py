from loveletter_ai.environment import LoveLetterEnv
from loveletter_ai.agents import ImprovedHeuristicAgent, RandomAgent
import random


def run_game(seed: int):
    env = LoveLetterEnv(players=["H", "R"], seed=seed)
    env.reset()

    agents = {
        0: ImprovedHeuristicAgent(),
        1: RandomAgent(),
    }

    rng = random.Random(seed)

    while env.state.phase != "round_over":
        pid = env.state.current_player

        obs = env.observe(pid)
        legal = env.legal_actions(pid)

        action = agents[pid].choose_action(obs, legal, rng)
        env.step(action)

    return env.state.winner


def batch_eval(total=1000):
    wins = 0

    for i in range(total):
        w = run_game(seed=i)

        if w is not None and int(w) == 0:
            wins += 1

        if (i + 1) % 100 == 0:
            print(f"{i+1}/{total} | winrate: {wins/(i+1):.2%}")

    print(f"\nFinal winrate: {wins/total:.2%}")


if __name__ == "__main__":
    batch_eval(1000)