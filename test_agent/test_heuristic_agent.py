from loveletter_ai.environment import GameEnv
from loveletter_ai.agents import ImprovedHeuristicAgent, RandomAgent

def batch_eval(total=1000):
    win = 0
    ag0 = ImprovedHeuristicAgent()
    ag1 = RandomAgent()
    for i in range(total):
        env = GameEnv([ag0, ag1])
        w,_ = env.run_game()
        if w == 0:
            win +=1
        if (i+1)%100 == 0:
            print(f"{i+1} | winrate:{win/(i+1):.2%}")
    print(f"Final win:{win/total:.2%}")

if __name__ == "__main__":
    batch_eval()
