# Regression test for wala/ML#435: a `@tf.function`-decorated Python function that
# returns a recursive call to itself used to drive
# `TensorGeneratorFactory.getGenerator` into unbounded recursion via the
# return-value follow-through and the assignment-graph predecessor walk, ending
# in `StackOverflowError`. The cycle guard added in this PR returns null
# (unknown) when a `PointsToSetVariable` is re-encountered along a single
# dispatch chain, so the analysis terminates instead of overflowing.
#
# Note: the runtime call below catches `RecursionError` because TensorFlow
# re-traces the recursive function on each call, hitting Python's default
# recursion limit. The static analysis path is what this test exercises.
import tensorflow as tf


def f(a):
    pass


@tf.function
def recursive_fn(n):
    if n > 0:
        return recursive_fn(n - 1)
    return n


try:
    result = recursive_fn(tf.constant(1))
    f(result)
except RecursionError:
    pass
