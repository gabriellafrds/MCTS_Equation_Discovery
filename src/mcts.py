import math
import random
from src.reward import compute_reward, scale_reward

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.N = 0
        self.Q = 0.0
        self.untried_actions = None

    def uct_score(self, c=1.0):
        if self.N == 0:
            return float('inf')
        return self.Q + c * math.sqrt(math.log(self.parent.N) / self.N)

    def best_child(self, c=1.0):
        return max(self.children, key=lambda child: child.uct_score(c))

    def is_fully_expanded(self, valid_actions):
        if self.untried_actions is None:
            self.untried_actions = list(valid_actions)
        return len(self.untried_actions) == 0

    def update(self, reward):
        self.N += 1
        self.Q += (reward - self.Q) / self.N

class MCTS:
    def __init__(self, grammar, data, locked_features_cols=None, c=1.0, 
                 n_simulations=10, t_max=50, gamma=0.01, alpha=0.005):
        self.locked_features_cols = locked_features_cols or []
        self.grammar = grammar
        self.data = data
        self.c = c
        self.n_simulations = n_simulations
        self.t_max = t_max
        self.gamma = gamma
        self.alpha = alpha

        self.best_reward = 0.0
        self.best_sequence = None
        self.global_max_reward = 1e-10

    def run(self, n_episodes):
        root = MCTSNode(state=[])
        for _ in range(n_episodes):
            node = self._select(root)
            node = self._expand(node)
            reward = self._simulate(node)
            self._backpropagate(node, reward)
        return self.best_sequence, self.best_reward

    def _select(self, node):
        while True:
            valid_actions = self.grammar.get_valid_actions(node.state)
            if not valid_actions or not node.is_fully_expanded(valid_actions):
                return node
            node = node.best_child(self.c)

    def _expand(self, node):
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
        Execute Sequential Halving (SHUSS) tournament over available actions.
        """
        if self.grammar.is_complete(node.state):
            r = compute_reward(node.state, self.data, self.grammar, 
                               self.locked_features_cols, self.gamma, self.alpha)
            self.global_max_reward = max(self.global_max_reward, r)
            if r > self.best_reward:
                self.best_reward, self.best_sequence = r, node.state
            return scale_reward(r, self.global_max_reward)

        valid_actions = self.grammar.get_valid_actions(node.state)
        if not valid_actions:
            return 0.0

        candidates = list(enumerate(valid_actions))
        stats = {i: {"sum": 0.0, "count": 0} for i, _ in candidates}
        n_rounds = max(1, math.ceil(math.log2(len(valid_actions))))
        best_round_reward = 0.0

        for r_idx in range(n_rounds):
            rollouts = max(1, self.n_simulations // (len(candidates) * n_rounds))
            
            for action_idx, action in candidates:
                for _ in range(rollouts):
                    seq = list(node.state) + [action]
                    
                    while len(seq) < self.t_max:
                        valid = self.grammar.get_valid_actions(seq)
                        if not valid:
                            break
                        seq.append(random.choice(valid))

                    if self.grammar.is_complete(seq):
                        r = compute_reward(seq, self.data, self.grammar, 
                                           self.locked_features_cols, self.gamma, self.alpha)
                        self.global_max_reward = max(self.global_max_reward, r)
                        scaled = scale_reward(r, self.global_max_reward)
                        
                        stats[action_idx]["sum"] += scaled
                        stats[action_idx]["count"] += 1
                        best_round_reward = max(best_round_reward, scaled)
                        
                        if r > self.best_reward:
                            self.best_reward, self.best_sequence = r, seq
            
            if len(candidates) > 1:
                averages = [(stats[idx]["sum"] / max(1, stats[idx]["count"]), (idx, action)) 
                            for idx, action in candidates]
                averages.sort(key=lambda x: x[0], reverse=True)
                candidates = [item[1] for item in averages[:math.ceil(len(candidates) / 2)]]

        return best_round_reward

    def _backpropagate(self, node, reward):
        while node is not None:
            node.update(reward)
            node = node.parent