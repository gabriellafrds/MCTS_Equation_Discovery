import math
import random
from grammar import RULES, NON_TERMINALS
from reward import compute_reward, scale_reward

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state  = state       # sequence of rules chosen so far
        self.parent = parent      # parent node in the MCTS tree

        self.children        = []     # list of child nodes already created
        self.N               = 0      # number of times this node was visited
        self.Q               = 0.0    # best reward seen through this node (max, not average)
        self.untried_actions = None   # actions not yet tried from this node

    def uct_score(self, c=1.0):
        """
        Compute the UCT score of this node as seen from its parent.
        Used to decide which child to select during the selection phase.
        """
        if self.N == 0:
            return float('inf')    # never visited -> absolute priority
        exploitation = self.Q
        exploration  = c * math.sqrt(math.log(self.parent.N) / self.N)
        return exploitation + exploration

    def best_child(self, c=1.0):
        """Return the child with the highest UCT score."""
        return max(self.children, key=lambda child: child.uct_score(c))

    def is_fully_expanded(self, valid_actions):
        """
        True if all possible children have been visited at least once.
        We initialize untried_actions on the first call.
        """
        if self.untried_actions is None:
            self.untried_actions = list(valid_actions)
        return len(self.untried_actions) == 0

    def update(self, reward):
        """
        Update statistics after a simulation.
        SPL uses max reward instead of average (greedy search).
        """
        self.N += 1
        self.Q  = max(self.Q, reward)

class MCTS:
    def __init__(self, grammar, data, c=1.0, n_simulations=10, t_max=50, eta=0.9999):
        self.grammar = grammar  # Grammar object (defines valid rules)
        self.data = data  # observed data (X, Y_dot)
        self.c = c  # exploration constant in UCT
        self.n_simulations = n_simulations  # number of random rollouts per expansion
        self.t_max = t_max  # maximum sequence length allowed
        self.eta = eta  # parsimony discount factor

        self.best_reward = 0.0  # best raw reward found so far across all episodes
        self.best_sequence = None  # sequence of rules that gave the best reward
        self.global_max_reward = 1e-10  # global maximum raw reward used for adaptive scaling

    def run(self, n_episodes):
        """
        Run n_episodes of MCTS and return the best equation found.
        Each episode consists of the 4 phases: select, expand, simulate, backpropagate.
        """
        root = MCTSNode(state=[])

        for _ in range(n_episodes):
            node   = self._select(root)          # phase 1: selection
            node   = self._expand(node)          # phase 2: expansion
            reward = self._simulate(node)        # phase 3: simulation
            self._backpropagate(node, reward)    # phase 4: backpropagation

        return self.best_sequence, self.best_reward

    def _select(self, node):
        """
        Phase 1 - Selection.
        Walk down the tree following the UCT policy until we reach
        a node that has untried actions (not fully expanded) or a
        terminal node (complete sequence, empty stack).
        """
        while True:
            valid_actions = self.grammar.get_valid_actions(node.state)

            # Terminal node: sequence is complete, no more rules to apply
            if not valid_actions:
                return node

            # Not fully expanded: stop here, expansion will handle it
            if not node.is_fully_expanded(valid_actions):
                return node

            # Fully expanded: go deeper following UCT
            node = node.best_child(self.c)

    def _expand(self, node):
        """
        Phase 2 - Expansion.
        Pick one untried action at random, create a new child node,
        and add it to the MCTS tree.
        """
        valid_actions = self.grammar.get_valid_actions(node.state)

        # Nothing to expand if sequence is already complete
        if not valid_actions:
            return node

        # Initialize untried actions if first time we expand this node
        if node.untried_actions is None:
            node.untried_actions = list(valid_actions)

        # Pick a random untried action and remove it from the list
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)

        # Create the new child node and attach it to the tree
        child = MCTSNode(state=node.state + [action], parent=node)
        node.children.append(child)
        return child

    def _simulate(self, node):
        """
        Phase 3 - Simulation (rollout).
        Complete the sequence randomly, evaluate, scale the reward,
        and return the best result across n_simulations rollouts.
        """
        best_reward = 0.0

        for _ in range(self.n_simulations):
            seq = list(node.state)

            while len(seq) < self.t_max:
                valid_actions = self.grammar.get_valid_actions(seq)
                if not valid_actions:
                    break
                seq.append(random.choice(valid_actions))

            if self.grammar.is_complete(seq):
                # Raw reward
                r = compute_reward(seq, self.data, self.grammar, eta=self.eta)

                # Update global max before scaling
                if r > self.global_max_reward:
                    self.global_max_reward = r

                # Scale reward (Equation 3 of the paper)
                r_scaled = scale_reward(r, self.global_max_reward)

                best_reward = max(best_reward, r_scaled)

                # Track best equation using raw reward for interpretability
                if r > self.best_reward:
                    self.best_reward = r
                    self.best_sequence = seq

        return best_reward

    def _backpropagate(self, node, reward):
        """
        Phase 4 - Backpropagation.
        Walk back up to the root, updating N and Q for every node
        along the path that was taken during selection.
        """
        while node is not None:
            node.update(reward)
            node = node.parent