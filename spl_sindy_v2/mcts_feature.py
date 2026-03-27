# spl_sindy_v2/mcts_feature.py
#
# MCTS agent that searches for ONE basis function at a time.
#
# This is the key architectural difference from the library-based approach:
# instead of building a full library in each episode, each MCTS run
# searches independently for the single best function phi_i(x) to add
# to the library.
#
# The MCTS logic (UCT, 4 phases, greedy Q=max) is identical to the
# original mcts.py. Only the reward function changes.

import math
import random
from spl_sindy_v2.reward_v2 import compute_reward, scale_reward


class MCTSNode:
    def __init__(self, state, parent=None):
        self.state           = state    # sequence of rules chosen so far
        self.parent          = parent   # parent in the MCTS tree
        self.children        = []       # child nodes created so far
        self.N               = 0        # visit count
        self.Q               = 0.0      # best reward seen (max, not avg)
        self.untried_actions = None     # actions not yet tried

    def uct_score(self, c=1.0):
        """UCT score for child selection. Returns inf if unvisited."""
        if self.N == 0:
            return float('inf')
        exploitation = self.Q
        exploration  = c * math.sqrt(math.log(self.parent.N) / self.N)
        return exploitation + exploration

    def best_child(self, c=1.0):
        """Return child with highest UCT score."""
        return max(self.children, key=lambda child: child.uct_score(c))

    def is_fully_expanded(self, valid_actions):
        """True if all possible children have been visited at least once."""
        if self.untried_actions is None:
            self.untried_actions = list(valid_actions)
        return len(self.untried_actions) == 0

    def update(self, reward):
        """Update N and Q — SPL greedy search: use max, not average."""
        self.N += 1
        self.Q  = max(self.Q, reward)


class MCTSFeature:
    """
    MCTS agent that searches for a single basis function phi(x).

    Identical to the original MCTS class except:
      - Uses reward_v2.compute_reward (R^2 based, no Powell)
      - Returns (best_sequence, best_reward) for one function
      - Can be run multiple times to build a library
    """

    def __init__(self, grammar, data, c=1.0, n_simulations=10,
                 t_max=15, eta=0.99):
        self.grammar       = grammar        # grammar namespace
        self.data          = data           # {"variables": ..., "y_dot": ...}
        self.c             = c              # UCT exploration constant
        self.n_simulations = n_simulations  # rollouts per expansion
        self.t_max         = t_max          # max sequence length
        self.eta           = eta            # parsimony discount

        self.best_reward       = 0.0
        self.best_sequence     = None
        self.global_max_reward = 1e-10      # for adaptive scaling

    def run(self, n_episodes):
        """
        Run n_episodes of MCTS and return the best single feature found.

        Returns (best_sequence, best_reward).
        """
        root = MCTSNode(state=[])

        for _ in range(n_episodes):
            node   = self._select(root)         # phase 1
            node   = self._expand(node)         # phase 2
            reward = self._simulate(node)       # phase 3
            self._backpropagate(node, reward)   # phase 4

        return self.best_sequence, self.best_reward

    def _select(self, node):
        """
        Phase 1 — Selection.
        Descend tree following UCT until a non-fully-expanded node.
        """
        while True:
            valid_actions = self.grammar.get_valid_actions(node.state)
            if not valid_actions:
                return node
            if not node.is_fully_expanded(valid_actions):
                return node
            node = node.best_child(self.c)

    def _expand(self, node):
        """
        Phase 2 — Expansion.
        Pick one untried action and create a new child node.
        """
        valid_actions = self.grammar.get_valid_actions(node.state)
        if not valid_actions:
            return node
        if node.untried_actions is None:
            node.untried_actions = list(valid_actions)

        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)

        child = MCTSNode(state=node.state + [action], parent=node)
        node.children.append(child)
        return child

    def _simulate(self, node):
        """
        Phase 3 — Simulation (rollout).
        Complete the sequence randomly n_simulations times.
        Keep the best reward (greedy SPL strategy).
        """
        best_reward = 0.0

        for _ in range(self.n_simulations):
            seq = list(node.state)

            # Complete randomly
            while len(seq) < self.t_max:
                valid_actions = self.grammar.get_valid_actions(seq)
                if not valid_actions:
                    break
                seq.append(random.choice(valid_actions))

            if self.grammar.is_complete(seq):
                # Raw reward (R^2 * parsimony)
                r = compute_reward(seq, self.data, self.grammar,
                                   eta=self.eta)

                # Adaptive scaling (SPL Eq. 3)
                if r > self.global_max_reward:
                    self.global_max_reward = r
                r_scaled    = scale_reward(r, self.global_max_reward)
                best_reward = max(best_reward, r_scaled)

                # Track global best
                if r > self.best_reward:
                    self.best_reward   = r
                    self.best_sequence = seq

        return best_reward

    def _backpropagate(self, node, reward):
        """
        Phase 4 — Backpropagation.
        Update N and Q from expanded node back to root.
        """
        while node is not None:
            node.update(reward)
            node = node.parent
