# Regression test for wala/ML#435: a recursive Python function whose return
# value flows back into itself used to drive
# `TensorGeneratorFactory.getGenerator` into unbounded recursion via the
# return-value follow-through and the assignment-graph predecessor walk,
# ending in `StackOverflowError`. The cycle guard added in this PR returns
# null (unknown) when a `PointsToSetVariable` is re-encountered along a single
# dispatch chain, so the analysis terminates and the base-case value
# (a scalar int32 tensor) flows back to `f`'s parameter.
import tensorflow as tf


def f(a):
    pass


def recursive_fn(n):
    if n > 0:
        return recursive_fn(n - 1)
    return n


result = recursive_fn(tf.constant(1))
assert isinstance(result, tf.Tensor)
assert result.shape == ()
assert result.dtype == tf.int32
f(result)
