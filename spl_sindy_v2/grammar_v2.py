# spl_sindy_v2/grammar_v2.py
#
# Grammar for single-feature MCTS search.
# Identical structure to the original grammar.py but:
#   - No C placeholder (SINDy finds coefficients)
#   - Supports multiple variables x, y, z for multi-dimensional systems
#   - No Library/Feature hierarchy — one expression per MCTS run
#
# This is used by MCTSFeature to search for one basis function at a time.

NON_TERMINALS = {"f", "M"}
TERMINALS     = {"x", "y", "z", "+", "*", "/", "-", "sin", "cos", "exp", "log"}

RULES = [
    ("f", ["M"]),

    ("M", ["M", "+", "M"]),     # addition        (binary)
    ("M", ["M", "-", "M"]),     # subtraction     (binary)
    ("M", ["M", "*", "M"]),     # multiplication  (binary)
    ("M", ["M", "/", "M"]),     # division        (binary)
    ("M", ["sin", "M"]),        # sine            (unary)
    ("M", ["cos", "M"]),        # cosine          (unary)
    ("M", ["exp", "M"]),        # exponential     (unary)
    ("M", ["log", "M"]),        # logarithm       (unary)
    ("M", ["x"]),               # variable x
    ("M", ["y"]),               # variable y
    ("M", ["z"]),               # variable z
    # No C — SINDy handles coefficients via STLSQ
]


def _get_stack(state):
    """
    Replay rule applications and return the current stack of
    non-terminals still waiting to be expanded. LIFO order.
    """
    stack = ["f"]
    for (left, right) in state:
        if not stack:
            break
        stack.pop(0)
        non_terminals_in_rule = [s for s in right if s in NON_TERMINALS]
        for symbol in reversed(non_terminals_in_rule):
            stack.insert(0, symbol)
    return stack


def get_valid_actions(state):
    """
    Return all rules applicable at the current state.
    Empty list means the expression is complete.
    """
    stack = _get_stack(state)
    if not stack:
        return []
    current_nt = stack[0]
    return [(left, right) for (left, right) in RULES
            if left == current_nt]


def is_complete(state):
    """Return True if the stack is empty (expression fully built)."""
    return len(_get_stack(state)) == 0


def get_grammar_for_variables(variable_names):
    """
    Dynamically build a grammar restricted to a given set of variables.
    Useful when the system has only x (Nguyen-1) or x,y,z (Lorenz).

    variable_names : list of strings, e.g. ["x"] or ["x", "y", "z"]

    Returns a SimpleNamespace-compatible grammar object.
    """
    import types

    # Build rules with only the requested variables
    base_rules = [
        ("f", ["M"]),
        ("M", ["M", "+", "M"]),
        ("M", ["M", "-", "M"]),
        ("M", ["M", "*", "M"]),
        ("M", ["M", "/", "M"]),
        ("M", ["sin", "M"]),
        ("M", ["cos", "M"]),
        ("M", ["exp", "M"]),
        ("M", ["log", "M"]),
    ]
    for var in variable_names:
        base_rules.append(("M", [var]))

    terminals = set(variable_names) | {"+", "-", "*", "/", "sin", "cos", "exp", "log"}

    def _stack(state):
        stack = ["f"]
        for (left, right) in state:
            if not stack:
                break
            stack.pop(0)
            nt_in_rule = [s for s in right if s in NON_TERMINALS]
            for sym in reversed(nt_in_rule):
                stack.insert(0, sym)
        return stack

    def _valid(state):
        stack = _stack(state)
        if not stack:
            return []
        nt = stack[0]
        return [(l, r) for (l, r) in base_rules if l == nt]

    def _complete(state):
        return len(_stack(state)) == 0

    grammar = types.SimpleNamespace(
        get_valid_actions = _valid,
        is_complete       = _complete,
        rules             = base_rules,
        non_terminals     = NON_TERMINALS,
        terminals         = terminals,
    )
    return grammar
