import tensorflow as tf


def f(a):
    pass


# `tf.random.truncated_normal(shape, ...)` produces a fresh tensor with the
# given shape and default `float32` dtype. See wala/ML#449: pre-fix this
# routed through `ReadDataFallback` (the per-class `isType` dispatch missed
# because `calledFunction` resolved to generic `LCodeBody`); post-fix
# `PROPERTY_NAME_GENERATORS` catches it via `truncated_normal` →
# `TruncatedNormal`.
result = tf.random.truncated_normal([2, 3])
assert isinstance(result, tf.Tensor)
assert result.shape == (2, 3)
assert result.dtype == tf.float32

f(result)
