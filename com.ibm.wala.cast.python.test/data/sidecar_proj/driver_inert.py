# Fixture for wala/ML#887: a sidecar entry that binds nothing. `t` is fully typed by inference,
# so the entry restating that same type refines no axis and contributes nothing. Before wala/ML#887
# that produced no diagnostic at all, so a correctly formed entry sitting beside a correct-looking
# result read as having produced it.
import tensorflow as tf


def consume(x):
    pass


t = tf.zeros((4, 3))
consume(t)
assert t.shape == (4, 3)
assert t.dtype == tf.float32
