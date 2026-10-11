# Test a `depth` whose points-to set holds None beside an integer: the parameter defaults to None
# and is overwritten before the call, so the None never reaches `one_hot` at run time.
import tensorflow as tf


def f(a):
    pass


def encode(labels, depth=None):
    if depth is None:
        depth = 3
    return tf.one_hot(labels, depth=depth)


arg = encode(tf.constant([0, 1, 2]))
assert isinstance(arg, tf.Tensor)
assert arg.dtype == tf.float32
assert arg.shape == (3, 3)

f(arg)
