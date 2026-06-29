# Love Letter Agent Ablation Results

Each row uses symmetric evaluation with 20 games per seat.
Profile: `quick`.
Win rate is for the first listed agent against RandomAgent unless stated otherwise.

## Fair Hidden-Information Agents vs Random

| Setting | Games | W-L-D | Win Rate | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Naive | 40 | 27-13-0 | 67.5% | 52.0%-79.9% |
| Logic | 40 | 31-9-0 | 77.5% | 62.5%-87.7% |
| Greedy | 40 | 31-9-0 | 77.5% | 62.5%-87.7% |
| Improved | 40 | 31-9-0 | 77.5% | 62.5%-87.7% |
| BeliefExpectimax | 40 | 27-13-0 | 67.5% | 52.0%-79.9% |
| BeliefMCTS | 40 | 27-13-0 | 67.5% | 52.0%-79.9% |
| Q-learning | 40 | 30-10-0 | 75.0% | 59.8%-85.8% |

## Ablation: BeliefExpectimax Samples

| Setting | Games | W-L-D | Win Rate | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| samples=1 | 40 | 31-9-0 | 77.5% | 62.5%-87.7% |
| samples=4 | 40 | 29-11-0 | 72.5% | 57.2%-83.9% |
| samples=8 | 40 | 26-14-0 | 65.0% | 49.5%-77.9% |

## Ablation: BeliefExpectimax Depth

| Setting | Games | W-L-D | Win Rate | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| depth=1 | 40 | 27-13-0 | 67.5% | 52.0%-79.9% |
| depth=2 | 40 | 27-13-0 | 67.5% | 52.0%-79.9% |

## Ablation: BeliefMCTS Simulations

| Setting | Games | W-L-D | Win Rate | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| simulations=4 | 40 | 25-15-0 | 62.5% | 47.0%-75.8% |
| simulations=12 | 40 | 23-17-0 | 57.5% | 42.2%-71.5% |
| simulations=24 | 40 | 29-11-0 | 72.5% | 57.2%-83.9% |

## Ablation: Q-learning Episodes

| Setting | Games | W-L-D | Win Rate | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| episodes=0 | 40 | 32-8-0 | 80.0% | 65.2%-89.5% |
| episodes=100 | 40 | 26-14-0 | 65.0% | 49.5%-77.9% |
| episodes=500 | 40 | 29-11-0 | 72.5% | 57.2%-83.9% |

## Oracle Baseline: Expectimax Depth

| Setting | Games | W-L-D | Win Rate | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| depth=1 | 40 | 30-10-0 | 75.0% | 59.8%-85.8% |
| depth=2 | 40 | 33-7-0 | 82.5% | 68.0%-91.2% |

## Oracle Baseline: MCTS Simulations

| Setting | Games | W-L-D | Win Rate | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| simulations=4 | 40 | 32-8-0 | 80.0% | 65.2%-89.5% |
| simulations=12 | 40 | 32-8-0 | 80.0% | 65.2%-89.5% |
