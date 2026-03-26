from grammar import RULES, NON_TERMINALS, TERMINALS

class Node:
    def __init__(self, symbol):
        self.symbol   = symbol
        self.children = []

    def is_terminal(self):
        # A node is terminal if its symbol is not a non-terminal
        return self.symbol not in NON_TERMINALS

    def is_complete(self):
        # A node is complete if it is terminal and all its children are complete
        if not self.is_terminal():
            return False
        return all(c.is_complete() for c in self.children)


def build_tree_step_by_step(sequence_of_rules):
    """
    Build an expression tree from a sequence of rules.
    Each rule is a tuple (left_side, right_side), e.g. ("M", ["M", "*", "M"]).
    This is the format used by mcts.py and grammar.py.

    Example:
        sequence_of_rules = [
            ("f", ["M"]),
            ("M", ["M", "*", "M"]),
            ("M", ["C"]),
            ("M", ["x"]),
        ]
        -> builds the tree for C * x
    """
    root  = Node("f")
    stack = [root]  # contains non-terminal nodes still waiting to be expanded

    for (left, right) in sequence_of_rules:  # unpack the tuple
        if not stack:
            break

        current_node = stack.pop(0)

        non_terminal_children = []
        for symbol in right:  # iterate over the right side of the rule
            child = Node(symbol)
            current_node.children.append(child)
            if symbol in NON_TERMINALS:
                non_terminal_children.append(child)

        # Insert in reverse order so the leftmost child ends up on top
        for child in reversed(non_terminal_children):
            stack.insert(0, child)

    return root


def extract_features_from_tree(node):
    """
    Extract a list of Feature nodes from a Library tree.
    """
    features = []
    
    if node.symbol == "f":
        if node.children:
            return extract_features_from_tree(node.children[0])
            
    elif node.symbol == "Library":
        # Library -> Feature
        # or Library -> Feature, Library
        if node.children:
            features.extend(extract_features_from_tree(node.children[0])) # The Feature
        if len(node.children) > 1:
            features.extend(extract_features_from_tree(node.children[1])) # The rest of the Library
            
    elif node.symbol == "Feature":
        # The child of Feature is M, which represents the math expression tree
        if node.children:
            features.append(node.children[0])
            
    return features