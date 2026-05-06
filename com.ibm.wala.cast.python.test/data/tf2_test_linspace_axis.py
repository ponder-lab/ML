import tensorflow as tf


def f(a):
    pass


# Drives the axis-passed branch in `Linspace.getDefaultShapes`. With axis=1 and
# vector start/stop, `tf.linspace` interpolates along axis 1, producing a
# higher-rank result whose precise shape (2, 5) requires combining `start`'s
# rank with `num` — the static analysis can't recover this, so the generator
# returns ⊤ (unknown shape).
start = tf.constant([0.0, 10.0])
stop = tf.constant([1.0, 20.0])
y = tf.linspace(start, stop, 5, axis=1)
assert isinstance(y, tf.Tensor)
assert y.shape == (2, 5)
assert y.dtype == tf.float32

f(y)
