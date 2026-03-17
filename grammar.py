# Production rules
# Each rule maps a non-terminal to a list of symbols
RULES = [
    ("f", ["M"]),

    ("M", ["M", "+", "M"]),
    ("M", ["M", "*", "M"]),
    ("M", ["M", "/", "M"]),
    ("M", ["sin", "M"]),
    ("M", ["cos", "M"]),
    ("M", ["exp", "M"]),
    ("M", ["x"]),
    ("M", ["y"]),
    ("M", ["C"]),
]

NON_TERMINALS = {"f", "M"}
TERMINALS = {"x", "y", "C", "+", "*", "/", "sin", "cos", "exp"}

def _get_stack(state):
    """
    Replay the rule applications in state and return the current
    stack of non-terminals still waiting to be expanded.

    This mirrors the LIFO mechanism in build_tree_step_by_step,
    but only tracks the stack without building actual tree nodes.

    Example:
        state = [("f", ["M"]), ("M", ["M", "+", "M"])]
        -> stack = ["M", "M"]  (two non-terminals left to expand)
    """
    # Start with the root non-terminal
    stack = ["f"]

    for (left, right) in state:
        if not stack:
            break

        # Remove the top non-terminal (it matches the left side of the rule)
        stack.pop(0)

        # Collect the non-terminals introduced by the right side of the rule
        non_terminals_in_rule = [s for s in right if s in NON_TERMINALS]

        # Insert them in reverse order so the leftmost one ends up on top
        for symbol in reversed(non_terminals_in_rule):
            stack.insert(0, symbol)

    return stack


def get_valid_actions(state):
    """
    Return all rules that can be applied at the current state.

    We look at the top of the stack to find the non-terminal
    that needs to be expanded next, then return all rules
    whose left side matches that non-terminal.

    Returns an empty list if the stack is empty (sequence is complete).
    """
    stack = _get_stack(state)

    # Stack is empty: no more rules can be applied
    if not stack:
        return []

    # The non-terminal on top of the stack is the one to expand next
    current_non_terminal = stack[0]

    # Return all rules that can replace this non-terminal
    return [(left, right) for (left, right) in RULES
            if left == current_non_terminal]


def is_complete(state):
    """
    Return True if the sequence of rules produces a complete expression,
    meaning the stack is empty (no non-terminals left to expand).
    """
    return len(_get_stack(state)) == 0
