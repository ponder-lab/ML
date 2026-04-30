import tensorflow as tf


def recursive_fn(n):
    if n > 0:
        return recursive_fn(n - 1)
    return 1


# `recursive_fn` is intentionally undecorated: the decorated form would re-trace
# the recursive call at runtime and AutoGraph would error on the
# `tf.Tensor`-as-Python-bool conversion before any assertions could run.
# The wala/ML#451 reproducer-2 regression doesn't depend on the decorator: the
# call-site argument is a real tensor either way, and the static analysis
# question (does `n - 1` get classified as a tensor only when `n` is a tensor?)
# is unchanged. The base case `return 1` is also kept — without it, this would
# duplicate `tf2_test_recursive_function.py`. Python execution stops at the
# recursive base case because the final return is a Python int, not a tensor;
# that's expected and parallels the issue body's fixture verbatim.
recursive_fn(tf.constant(5))
