# Production rules
# Each rule maps a non-terminal to a list of symbols
RULES = [
    ("f", ["M"]),
    ("M", ["M", "*", "M"]),
    ("M", ["sin", "M"]),
    ("M", ["cos", "M"]),
    ("M", ["exp", "M"]),
    ("M", ["-", "M"]),
    ("M", ["x"]),
    ("M", ["y"]),
    ("M", ["z"]),
    ("M", ["1"])
]

NON_TERMINALS = {"f", "M"}
TERMINALS = {"x", "y", "z", "*", "sin", "cos", "exp", "-", "1"}

def _get_stack(state):
    """
    Replay rule applications to determine the stack of non-terminals left to expand.
    """
    # Start with the root non-terminal
    stack = ["f"]
    for left, right in state:
        if not stack:
            break

        # remove the top non-terminal (it matches the left side of the rule)
        stack.pop(0)
        # Push new non-terminals to the front in order
        stack = [s for s in right if s in NON_TERMINALS] + stack
    return stack

def get_valid_actions(state):
    """
    Return all valid grammar rules applicable at the current state.
    """
    stack = _get_stack(state)

    # stack is empty -> no more rules can be applied
    if not stack:
        return []
    
    # the non-terminal on top of the stack is the one to expand next
    current_non_terminal = stack[0]

    # Return all rules that can replace this non-terminal
    return [(left, right) for (left, right) in RULES
            if left == current_non_terminal]


def is_complete(state):
    """
    Return True if all branches have resolved to terminal symbols.
    """
    return len(_get_stack(state)) == 0
