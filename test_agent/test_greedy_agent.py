from loveletter_ai.environment import GameEnv
from loveletter_ai.agents import GreedyAgent, RandomAgent
import random

def single_test():
    agents = [GreedyAgent("Greedy"), RandomAgent("Rand")]
    env = GameEnv(agents)
    winner,_ = env.run_game()
    print(f"Single game winner:{winner}")

def batch_eval(total=1000):
    win = 0
    ag0 = GreedyAgent()
    ag1 = RandomAgent()
    for idx in range(total):
        env = GameEnv([ag0, ag1])
        w,_ = env.run_game()
        if w == 0:
            win +=1
        if (idx+1)%100==0:
            print(f"{idx+1}/{total} | WinRate:{win/(idx+1):.2%}")
    print(f"\nFinal WinRate:{win/total:.2%}")

if __name__=="__main__":
    batch_eval(1000)
