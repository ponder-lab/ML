import tensorflow as tf


def f(a):
    pass


# `tf.where(condition, x, y)` with broadcast-compatible different shapes:
# `x` is `(3,)` and `y` is `(2, 3)`, so the runtime result broadcasts to
# `(2, 3)`. This fixture exercises the union path in the dedicated `Where`
# generator: the static analysis unions `x`'s shape `{(3,)}` with `y`'s
# shape `{(2, 3)}` to produce `{(3,), (2, 3)}`. This is sound but
# imprecise &mdash; the precise answer (`{(2, 3)}`, the broadcast result)
# is deferred to wala/ML#482's broadcast-composition follow-up.
condition = tf.constant([True, False, True])
x = tf.constant([1.0, 2.0, 3.0])
y = tf.constant([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
assert isinstance(x, tf.Tensor)
assert x.shape == (3,)
assert x.dtype == tf.float32
assert isinstance(y, tf.Tensor)
assert y.shape == (2, 3)
assert y.dtype == tf.float32
result = tf.where(condition, x, y)
assert isinstance(result, tf.Tensor)
assert result.shape == (2, 3)
assert result.dtype == tf.float32
f(result)
