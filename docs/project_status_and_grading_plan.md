# Project Status and Grading Plan

## 1. Alignment with the Proposal

The proposal defines the project as an imperfect-information AI study on Love
Letter, with four implementation families and three evaluation directions.
The current codebase is broadly aligned with that plan, but the report and
presentation should be careful about which parts are complete experiments and
which parts are still approximations.

| Proposal item | Current status | Evidence in code | Notes for report |
| --- | --- | --- | --- |
| Game background: 16 cards, draw 1/play 1, win by elimination or showdown | Implemented | `cards.py`, `environment.py` | The environment implements the base 16-card deck and core card effects. |
| Visible/hidden state split | Implemented | `GameState`, `Observation`, `LoveLetterEnv.observe` | This is one of the strongest course-relevance points: hidden information is enforced by observation construction. |
| Stochastic transitions from draws | Implemented | seeded deck shuffle, draw phase, hidden card | Deterministic under seed, stochastic over seeds. |
| Heuristic Agent | Implemented and improved | `NaiveHeuristicAgent`, `GreedyAgent`, `ImprovedHeuristicAgent` | Improved version uses card-count-based scoring and avoids impossible Guard guesses. |
| Bayesian Agent | Partially implemented | `BayesianBelief`, `BeliefExpectimaxAgent` | It samples plausible hidden states from observations and logic constraints; it is not a full Bayesian posterior over complete histories. |
| Expectimax Agent | Implemented as deterministic search baseline | `ExpectimaxAgent` | Strong performance, but it searches cloned true state when called through `choose_action_from_env`. Explain this as a controlled baseline, not fully fair hidden-information play. |
| RL and MCTS | Implemented and split by fairness | `QLearningAgent`, `ApproximateQLearningAgent`, `BeliefMCTSAgent`, `MCTSAgent`, `train_q_learning_mixed` | RL now includes decaying epsilon, alpha decay, mixed opponents, reward shaping, and Q-learning variant ablations. `BeliefMCTSAgent` is the fair hidden-information version; `MCTSAgent` is an oracle baseline. |
| Tournament matrix | Partially implemented | `evaluate_pair`, `evaluate_pair_symmetric`, `evaluate_agents.py` | Benchmark output now separates fair hidden-information agents from oracle baselines; still need a full all-pairs matrix for the final report. |
| Robustness tests | Partially implemented | deterministic seed tests and symmetric evaluation | Need more explicit robustness experiments over seeds, seats, and search budgets. |
| Ablation study | Runner implemented | `run_ablation_experiments.py`, `outputs/experiments/` | The quick display tables exist; final report should rerun with larger sample sizes. |

## 2. Current Implementation Progress

### Environment and Game Engine

Implemented:
- Full base Love Letter deck with card value/count metadata.
- Two to four players supported by the environment constructor.
- Deterministic reset under explicit seed.
- Draw phase, action phase, elimination, showdown, and winner handling.
- Legal action generation, including Countess forced play, Guard restrictions,
  protected targets, and Prince self-targeting.
- Public/private event history and per-player observations.
- Terminal renderer for debugging with masked or revealed state.

Why it matters for CS181:
- The environment is a clean model of a stochastic, partially observable,
  adversarial decision problem.
- Agents receive observations and legal actions rather than mutating true state.
- Search and learning methods can be compared under reproducible seeds.

### Agent Families

Implemented:
- `RandomAgent`: uniform random baseline.
- `NaiveHeuristicAgent`: simple handcrafted rule baseline.
- `GreedyAgent` and `ImprovedHeuristicAgent`: card-count-aware heuristic
  policies. They avoid impossible Guard guesses and score Baron/Prince/King
  using retained-card value and unseen-card counts.
- `LogicAgent`: maintains symbolic possible-card sets from public/private
  events; uses deterministic Guard guesses when a target card is known.
- `ExpectimaxAgent`: depth-limited deterministic expectimax over cloned
  environment states.
- `BayesianBelief` and `BeliefExpectimaxAgent`: approximate hidden-state
  sampling plus expectimax over sampled determinizations.
- `QLearningAgent`: tabular Q-learning with heuristic prior, decaying
  exploration, and per-state alpha decay.
- `ApproximateQLearningAgent`: feature-based linear Q-learning with
  reward-shaping support and monotone tactical feature constraints.
- `BeliefMCTSAgent`: information-set MCTS with root hidden-state sampling.
- `MCTSAgent`: perfect-information MCTS with cloned-environment rollouts, kept
  as an oracle-style upper-bound baseline.

Current interpretation:
- The strongest "AI reasoning" story is the progression:
  Random -> Naive heuristic -> Logic/card counting -> Belief sampling ->
  Expectimax/MCTS planning -> repaired Q-learning and approximate Q-learning.
- The most honest framing is that this is a compact research environment plus
  several baseline agents, not a solved Love Letter bot.

### Evaluation and Tests

Implemented:
- Unit tests for deterministic setup, rendering privacy, belief sampling,
  knowledge-base deductions, Q-learning update, MCTS/Expectimax legal actions,
  and integrated match completion.
- `evaluate_pair`: fixed-seat evaluation.
- `evaluate_pair_symmetric`: both agents play both seats to reduce seat bias.
- `test_agent/evaluate_agents.py`: quick benchmark vs Random, split into fair
  hidden-information agents and oracle baselines.
- `test_agent/run_ablation_experiments.py`: writes CSV and Markdown ablation
  tables under `outputs/experiments/`.

Latest local checks:
- `python -m pytest -q`: 19 tests passed.
- `python test_agent/evaluate_agents.py 10`: fair/oracle benchmark smoke test
  completed.
- `python test_agent/run_ablation_experiments.py 5 outputs/experiments quick`:
  generated `outputs/experiments/ablation_results.csv` and
  `outputs/experiments/ablation_results.md`.

These quick numbers are useful as a sanity check, but the final report should
use larger repeated trials and confidence intervals.

Current experiment commands:

```bash
python test_agent/evaluate_agents.py 25
python test_agent/run_ablation_experiments.py 5 outputs/experiments quick
```

For final report-grade numbers, use a larger run:

```bash
python test_agent/run_ablation_experiments.py 50 outputs/experiments full
```

The quick profile is meant to demonstrate table format and code path only. Its
confidence intervals are wide and should not be treated as final evidence.

## 3. Main Shortcomings and Risk Areas

### 1. Fairness of search agents

`ExpectimaxAgent` and `MCTSAgent` can access cloned true environment state in
`choose_action_from_env`. This is useful as a planning baseline, but it is not
fully fair under hidden information. If the presentation claims these agents
operate only on observations, that would be unsound.

Recommended framing:
- Call them "perfect-information / determinized planning baselines."
- Present `BeliefExpectimaxAgent` and `BeliefMCTSAgent` as the fair
  hidden-information search agents.
- Keep oracle search results in a separate upper-bound table.

### 2. Bayesian model is approximate

The belief module samples hidden hands and deck order from remaining card
counts plus simple logical constraints. It does not compute a full posterior
over all histories and opponent policies.

Recommended framing:
- Describe it as an approximate Bayesian belief sampler.
- Explain what evidence it uses: own hand, discards, deck size, protected or
  eliminated status, Priest observations, failed Guard guesses.
- Avoid claiming exact Bayesian inference.

### 3. Q-learning needed repairs and ablations

The original Q-table used compact observation/action keys and terminal rewards.
This was easy to explain, but it suffered from sparse state coverage and noisy
updates. The repaired version keeps the tabular method as a baseline and adds:
- decaying epsilon from exploration to exploitation;
- per-state alpha decay to reduce single-game noise;
- mixed opponents (`Random`, `Naive`, `Improved`, and self-play);
- reward shaping for eliminations, Princess mistakes, Guard hits, Priest
  information, and Princess protection;
- an approximate Q-learning agent over interpretable action features;
- explicit ablations for heuristic-only, pure Q, Q+prior, and approximate Q.

Recommended additions:
- Show learning curves vs training episodes with larger confidence intervals.
- Report tabular Q-table size, approximate feature weights, and win rate after
  0, 500, 1000, 5000, and 10000 episodes.
- Explain why pure tabular Q remains weak: partial observability, sparse
  terminal reward, and non-stationary opponent behavior.

### 4. Evaluation is not yet report-grade

The current benchmark script is good for development, but the proposal promises
tournament matrix, robustness test, and ablation study. Those should become
real tables/figures.

Remaining report-grade experiments:
- Full pairwise tournament matrix: all agents vs all agents, symmetric seats.
- Robustness: repeat across many seed ranges; report mean and 95% confidence
  interval.
- Larger ablation run:
  - Naive vs Improved heuristic.
  - Logic enabled vs fallback only.
  - BeliefExpectimax samples.
  - BeliefMCTS simulations.
  - Oracle Expectimax depth.
  - Oracle MCTS simulations.
  - Q-learning variants and approximate Q-learning episodes.

### 5. Documentation is partly stale

Some early docs still describe the project as a skeleton or refer to
`agents.py`, while the code has been refactored into `loveletter_ai/agents/`.
Before submission, update README and docs so graders do not see inconsistency.

### 6. Scope needs to be explicit

The environment supports 2-4 players, but most evaluation currently focuses on
2-player games. This is acceptable if stated clearly. If time permits, add a
small 3-player/4-player smoke test or explain that experiments use the 2-player
setting for controlled comparison.

### 7. Reward definition differs from the proposal

The proposal says `R=+1` for victory and `R=0` for elimination or loss. The
current Q-learning helper uses `+1` for a win and `-1` for a loss, plus small
training-only shaping rewards. This is a reasonable reinforcement-learning
choice because it separates bad terminal outcomes from neutral transitions and
helps assign credit in a sparse game. The report should explicitly state this
as an implementation refinement relative to the proposal.

## 4. High-Score Repair Plan

### Must Do

1. Update README and report text so they match the current package structure.
2. Use the reproducible ablation runner to generate final CSV/Markdown results.
3. Generate a full pairwise tournament matrix with symmetric seats.
4. Add confidence intervals or standard errors for win rates.
5. Include the ablation tables from `outputs/experiments/`.
6. Clearly distinguish perfect-information baselines from hidden-information
   agents.
7. Add an "External resources and tools" section to the report and final slide.

### Should Do

1. Use `BeliefMCTSAgent` as the MCTS result in the fair hidden-information
   table, and keep `MCTSAgent` only in the oracle baseline table.
2. Add learning curve plots for approximate Q-learning.
3. Add runtime/performance discussion: depth/samples/simulations trade off
   strength against compute time.
4. Add a short limitations slide: approximate belief, uniform opponent model,
   tabular RL sparsity, mostly 2-player evaluation.
5. Add comments/docstrings for algorithmic choices where graders will inspect
   code.

### Nice to Have

1. Serialize match logs for qualitative examples.
2. Show one example belief distribution after public observations.
3. Add 3-player/4-player evaluation smoke tests.
4. Add a simple chart-generation script for report figures.

## 5. Suggested Final Report Structure

1. Introduction and motivation
   - Love Letter as a small partially observable stochastic game.
   - Course relevance: search, uncertainty, inference, RL.
2. Game model
   - State, observation, action, transition, reward.
   - Explicit hidden vs visible information.
3. Environment implementation
   - Deterministic seeded engine, legal actions, card effects, observations.
4. Agents
   - Random and Naive baselines.
   - Improved heuristic/card counting.
   - Logic knowledge base.
   - Expectimax perfect-information baseline.
   - BeliefExpectimax approximate hidden-state search.
   - Q-learning and MCTS.
5. Experiments
   - Protocol: symmetric seats, seeds, number of games, metrics.
   - Tournament matrix.
   - Robustness over seeds.
   - Ablations.
6. Results and analysis
   - Which agents improve over baselines.
   - Where search helps.
   - Where hidden information limits performance.
   - Why Q-learning is hard.
7. Limitations and future work
   - Full posterior inference, opponent modeling, belief-MCTS, larger RL.
8. External resources and tools
   - Code/libraries/tools used and why.

## 6. Suggested Presentation Story

Slide 1: Problem and motivation
- "Love Letter is a compact imperfect-information game."

Slide 2: Formal model
- Visible state, hidden state, actions, stochastic transition, reward.

Slide 3: Environment architecture
- True state -> observation -> agent -> legal action -> transition.

Slide 4: Agents
- Show the ladder from Random to heuristic, logic, belief search, RL/MCTS.

Slide 5: Evaluation protocol
- Symmetric tournament, robustness, ablations.

Slide 6: Results
- Tournament matrix and one key ablation chart.

Slide 7: Analysis
- What improved, what failed, and why hidden information matters.

Slide 8: Limitations and external resources
- Be explicit and transparent.

## 7. External Resources and Tools Disclosure

Use this style in the final report and final presentation:

| Resource/tool | How it was used | Why it was appropriate |
| --- | --- | --- |
| Python standard library (`random`, `dataclasses`, `enum`, `copy`, `collections`) | Implemented deterministic environment, data structures, cloning, counters, and seeded randomness. | No external game engine was needed; standard library keeps the project inspectable. |
| `pytest` | Ran unit and integration tests. | Verifies correctness of setup, hidden information masking, agent decisions, and match completion. |
| Codex / AI coding assistant | Helped refactor branches, identify bugs, write tests, and organize documentation. | Used as a development tool; final code and explanations should be reviewed by the team. |
| Love Letter base rules | Encoded the 16-card deck and card effects. | Defines the game domain; no external Love Letter engine code was copied. |

If any team member used online articles, code snippets, libraries, or generated
assets beyond this list, add them explicitly. The safest high-score position is:
"We wrote the game engine and agents ourselves; external tools were used for
testing, development assistance, and presentation/report preparation."
