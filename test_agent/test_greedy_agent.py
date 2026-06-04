from loveletter_ai.environment import LoveLetterEnv
from loveletter_ai.agents import GreedyAgent, RandomAgent
import random


def run_single(seed=0):
    env = LoveLetterEnv(players=["Greedy", "Random"], seed=seed)

    agents = {
        0: GreedyAgent(),
        1: RandomAgent(),
    }

    rng = random.Random(seed)

    env.reset()

    while env.state.phase != "round_over":
        pid = env.state.current_player

        obs = env.observe(pid)
        legal = env.legal_actions(pid)

        action = agents[pid].choose_action(obs, legal, rng)
        env.step(action)

    return env.state.winner


def batch_eval(total=1000):
    win = 0

    for i in range(total):
        w = run_single(seed=i)

        if w is not None and int(w) == 0:
            win += 1

        if (i + 1) % 100 == 0:
            print(f"{i+1}/{total} | WinRate: {win/(i+1):.2%}")

    print(f"\nFinal WinRate: {win/total:.2%}")


if __name__ == "__main__":
    batch_eval(1000)