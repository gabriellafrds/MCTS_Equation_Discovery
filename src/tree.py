from src.grammar import NON_TERMINALS

class Node:
    def __init__(self, symbol):
        self.symbol   = symbol
        self.children = []

    def is_terminal(self):
        # a node is terminal if its symbol is not a non-terminal
        return self.symbol not in NON_TERMINALS

    def is_complete(self):
        # a node is complete if it is terminal and all its children are complete
        if not self.is_terminal():
            return False
        return all(c.is_complete() for c in self.children)


def build_tree_step_by_step(sequence_of_rules):
    """
    Build an expression tree from a sequence of rules.
    Each rule is a tuple (left_side, right_side), e.g. ("M", ["M", "*", "M"]).
    This is the format used by mcts.py and grammar.py.

    example:
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
        
        # Instantiate children directly
        new_children = [Node(symbol) for symbol in right]
        current_node.children.extend(new_children)

        # Prepend new non-terminal sub-nodes to the expansion stack
        stack[:0] = [child for symbol, child in zip(right, new_children) if symbol in NON_TERMINALS]

    return root


def extract_features_from_tree(node):
    """
    Extract all Feature nodes logically from a generalized Library AST.
    """
    features = []
    
    if node.symbol == "f" and node.children:
        return extract_features_from_tree(node.children[0])
            
    elif node.symbol == "Library":
        if node.children:
            features.extend(extract_features_from_tree(node.children[0])) # The Feature
        if len(node.children) > 1:
            features.extend(extract_features_from_tree(node.children[1])) # The rest of the Library
            
    elif node.symbol == "Feature" and node.children:
        features.append(node.children[0])
            
    return features

def count_nodes(node):
    """
    Recursively count the total number of rule modules in an AST.
    """
    return 1 + sum(count_nodes(child) for child in node.children)