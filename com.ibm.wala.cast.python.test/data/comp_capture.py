# Names an enclosed function reads from its enclosing function: a lambda's, a nested function's, a
# comprehension element's and a comprehension filter's.


def via_lambda(x):
    f = lambda: x + 1
    return f()


def via_nested_def(x):
    def f():
        return x + 1

    return f()


def via_comprehension(x):
    return [x + k for k in (1, 2)]


def via_filter(x):
    return [k for k in (1, 2) if k > x]


via_lambda(1)
via_nested_def(1)
via_comprehension(1)
via_filter(1)
