# Q-learning and MCTS Agents for Love Letter

This document describes how our AI agent applies Q-learning and Monte Carlo Tree
Search (MCTS) in the Love Letter environment. The writing style follows a
report-oriented structure: first defining the state, action, reward, and
transition, then explaining the algorithmic procedure and update formulas.

## Problem Formulation

Love Letter is an imperfect-information card game. At each decision point, a
player observes public information, their own hand, and part of the event
history, but not the opponent's hidden hand or the remaining deck order.

We define one transition as

$$
(s, a, s')
$$

where \(s\) is the current observation/state representation, \(a\) is the legal
action selected by the agent, and \(s'\) is the next state after the environment
resolves the action and card effect.

In our implementation, the environment provides all legal actions:

$$
A(s) = \{a_1, a_2, ..., a_n\}
$$

The agent only chooses from \(A(s)\), so rules such as Countess forced play,
Handmaid protection, valid Guard guesses, and target restrictions are enforced
by the environment.

## State Representation

For Q-learning, the full hidden game state is not used directly. Instead, the
agent uses an observation-based feature tuple:

$$
s =
(\text{phase}, \text{current player}, \text{deck size}, H_{\text{self}},
P_{\text{public}})
$$

where:

- \(H_{\text{self}}\) is the current player's visible hand.
- \(P_{\text{public}}\) includes each player's hand size, discard pile,
  elimination status, and protection status.
- Hidden opponent cards, the hidden set-aside card, and exact deck order are not
  included.

This makes the RL state compatible with the information actually available to
the player.

For MCTS, the agent uses the cloned environment for simulation. In the first
version, this can be a determinized search from the current environment state.
For a stricter imperfect-information version, the agent samples possible hidden
states that are consistent with the root player's observation.

## Action Representation

Each action is represented by:

$$
a = (\text{card}, \text{target}, \text{guess})
$$

Examples:

- Play Handmaid: \((\text{Handmaid}, \varnothing, \varnothing)\)
- Play Guard on player 1 and guess Princess:
  \((\text{Guard}, 1, \text{Princess})\)
- Play Prince targeting self:
  \((\text{Prince}, \text{self}, \varnothing)\)

The environment returns only legal actions, so the agent does not need to
manually filter invalid moves.

## Evaluation Function

Although Q-learning mainly learns from rewards, a simple state evaluation
function is useful for MCTS rollouts and for optional reward shaping. We define:

$$
f(s, i) =
w_1 I_i
+ w_2 C_i
+ w_3 D_i
+ w_4 P_i
+ w_5 T_i
$$

where \(i\) is the player being evaluated.

The features are:

- \(I_i\): alive indicator. It is \(1\) if player \(i\) is alive, otherwise
  \(-1\).
- \(C_i\): current hand card value.
- \(D_i\): total value of cards in player \(i\)'s discard pile.
- \(P_i\): protection indicator from Handmaid.
- \(T_i\): opponent elimination advantage.

One possible concrete version is:

$$
f(s, i)
= 100 I_i
+ 10 \cdot \text{handValue}_i
+ \text{discardTotal}_i
+ 8 \cdot \text{protected}_i
+ 30 \cdot N_{\text{opponents eliminated}}
$$

This function rewards survival, keeping high-value cards, having useful discard
tie-breakers, being protected, and eliminating opponents.

## Reward Design for Q-learning

The simplest reward is sparse terminal reward:

$$
R(s, a, s') =
\begin{cases}
1, & \text{if the agent wins the round} \\
-1, & \text{if the agent loses the round} \\
0, & \text{otherwise}
\end{cases}
$$

This reward is easy to interpret but slow to learn because most intermediate
actions receive reward \(0\).

We can also define a shaped reward using the evaluation function:

$$
R(s, a, s') = f(s', i) - f(s, i)
$$

This gives positive reward when the action improves the player's position, for
example by eliminating an opponent, protecting the Princess with Handmaid, or
keeping a higher-value card for showdown.

To account for opponent improvement, we can use a relative reward:

$$
R(s, a, s') =
\left[f(s', i) - f(s, i)\right]
-
\lambda
\sum_{j \neq i}
\left[f(s', j) - f(s, j)\right]
$$

where \(\lambda\) controls how much the agent penalizes improvements in
opponents' positions.

For the first implementation, we use sparse terminal reward. After the baseline
works, shaped reward can be added to speed up learning.

## Q-learning Agent

Q-learning learns a value \(Q(s,a)\), which estimates the expected future return
after taking action \(a\) in state \(s\).

At each turn:

1. The environment gives the current observation \(s\).
2. The environment gives legal actions \(A(s)\).
3. The agent chooses one action \(a \in A(s)\).
4. The environment applies the action and reaches \(s'\).
5. The agent receives reward \(r = R(s,a,s')\).
6. The agent updates \(Q(s,a)\).

The update target is:

$$
\text{sample}
= r + \gamma \max_{a' \in A(s')} Q(s', a')
$$

Then the Q-value is updated by:

$$
Q(s,a)
\leftarrow
(1-\alpha)Q(s,a) + \alpha \cdot \text{sample}
$$

Equivalently:

$$
Q(s,a)
\leftarrow
Q(s,a)
+ \alpha
\left[
r + \gamma \max_{a' \in A(s')} Q(s',a') - Q(s,a)
\right]
$$

where:

- \(\alpha\) is the learning rate.
- \(\gamma\) is the discount factor.
- \(r\) is the immediate reward.
- \(\max_{a'} Q(s',a')\) estimates the best future value.

## Q-learning Action Selection

During training, the agent uses \(\epsilon\)-greedy exploration:

$$
\pi(s) =
\begin{cases}
\text{random action from } A(s), & \text{with probability } \epsilon \\
\arg\max_{a \in A(s)} Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

This allows the agent to explore unfamiliar actions early in training while
gradually exploiting learned high-value actions.

During evaluation, we set \(\epsilon = 0\), so the agent always chooses:

$$
a^* = \arg\max_{a \in A(s)} Q(s,a)
$$

If multiple legal actions have the same Q-value, the tie is broken randomly
using the environment's seeded random number generator.

## Q-learning Training Procedure

The training process is self-play or training against a baseline agent.

For each episode:

1. Initialize a new Love Letter round.
2. For each turn, observe \(s\) and legal actions \(A(s)\).
3. Select action \(a\) using \(\epsilon\)-greedy.
4. Apply \(a\) in the environment.
5. Observe next state \(s'\) and reward \(r\).
6. Update \(Q(s,a)\).
7. Continue until the round ends.

At the end of the round, the final reward is assigned:

$$
r_{\text{terminal}} =
\begin{cases}
1, & \text{winner is the learning agent} \\
-1, & \text{otherwise}
\end{cases}
$$

The Q-table is stored as:

$$
Q[(\text{state key}, \text{action key})] = \text{value}
$$

Unvisited state-action pairs have default value:

$$
Q(s,a) = 0
$$

## Monte Carlo Tree Search Agent

MCTS chooses an action by building a partial search tree from the current state.
Instead of evaluating every possible future exactly, it estimates action quality
through repeated simulations.

Each node \(v\) in the tree stores:

- \(N(v)\): number of visits.
- \(Q(v)\): accumulated simulation value.
- parent node.
- children nodes.
- the action that led to this node.

Each MCTS iteration has four steps:

1. Selection
2. Expansion
3. Simulation
4. Backpropagation

## MCTS Step 1: Selection

Starting from the root node, the agent repeatedly selects the child with the
largest UCT value:

$$
\text{UCT}(v')
=
\frac{Q(v')}{N(v')}
+
c
\sqrt{
\frac{2 \ln N(v)}{N(v')}
}
$$

where:

- \(v\) is the parent node.
- \(v'\) is a child node.
- \(Q(v') / N(v')\) is the average simulation value.
- \(c\) is the exploration constant.

The first term exploits actions that already performed well. The second term
encourages exploration of less-visited actions.

## MCTS Step 2: Expansion

When the selected node is not terminal and still has untried legal actions, the
agent expands one new child.

For the selected state \(s\), the environment computes:

$$
A(s) = \{a_1, a_2, ..., a_n\}
$$

For an untried action \(a_k\), the agent clones the environment, applies
\(a_k\), and creates a child node representing the resulting state.

## MCTS Step 3: Simulation

From the expanded child, the agent simulates a game until either:

- the round ends, or
- a rollout depth limit is reached.

During rollout, actions can be selected by a simple policy:

$$
\pi_{\text{rollout}}(s) =
\begin{cases}
\text{heuristic action}, & \text{for the root player} \\
\text{random legal action}, & \text{for opponents}
\end{cases}
$$

If the rollout reaches the end of the round, the simulation value is:

$$
z =
\begin{cases}
1, & \text{root player wins} \\
-1, & \text{root player loses} \\
0, & \text{no winner}
\end{cases}
$$

If the rollout stops because of the depth limit, the value can be approximated
by the evaluation function:

$$
z = \text{clip}\left(\frac{f(s, i)}{100}, -1, 1\right)
$$

## MCTS Step 4: Backpropagation

After simulation, the value \(z\) is propagated back through every node on the
selected path.

For each node \(v\) on the path:

$$
N(v) \leftarrow N(v) + 1
$$

$$
Q(v) \leftarrow Q(v) + z
$$

Thus, frequently visited actions with high average rollout value become more
likely to be selected in later iterations.

## Final MCTS Action Choice

After a fixed number of simulations, the agent chooses the root action with the
largest visit count:

$$
a^*
=
\arg\max_{a \in A(s)}
N(\text{child}(s,a))
$$

If two actions have the same visit count, the tie is broken by average value:

$$
\frac{Q(\text{child}(s,a))}{N(\text{child}(s,a))}
$$

## Imperfect Information Handling

Love Letter contains hidden cards, so the MCTS agent cannot assume it knows the
true state when making a decision from a player observation.

To handle this, we use determinization. At the root of search:

1. Keep the root player's known hand fixed.
2. Keep public discard piles fixed.
3. Build the set of unseen cards from the full deck minus visible cards.
4. Randomly sample possible opponent hands from unseen cards.
5. Randomly sample the remaining deck and hidden set-aside card.
6. Run MCTS on this sampled complete state.

The agent can repeat this process across multiple sampled states and average the
root action statistics.

This approximates decision-making under hidden information while still allowing
the standard MCTS procedure to run on complete game states.

## Comparison Between Q-learning and MCTS

Q-learning learns from many past episodes. Its decision rule becomes fast after
training because it only needs to look up:

$$
\arg\max_{a \in A(s)} Q(s,a)
$$

However, tabular Q-learning may suffer from many unvisited states because Love
Letter has stochastic draws and hidden information.

MCTS does not require offline training. It searches online from the current
state and estimates action quality by simulation. Its strength depends on the
number of simulations and the quality of the rollout policy, but inference is
slower than a trained Q-learning table.

In our project, Q-learning provides a learned RL baseline, while MCTS provides a
search-based planning agent. Comparing them helps evaluate the tradeoff between
offline learning and online simulation.
