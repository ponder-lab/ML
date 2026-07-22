# Test wala/ML#734: a constant transpose permutation fixes its unresolved input's
# rank. `t` is shape-opaque with a proven dtype (a cast of a `tf.roll` result the
# analysis does not model), and `f`'s explicit-perm transpose proves `inputs`
# rank-3 (the `crf_forward` pattern).
import tensorflow as tf


def f(inputs):
    assert inputs.shape == (4, 3, 5)
    assert inputs.dtype == tf.float32
    x = tf.transpose(inputs, [1, 0, 2])
    assert x.shape == (3, 4, 5)
    return x


t = tf.cast(tf.roll(tf.ones((4, 3, 5)), shift=1, axis=0), tf.float32)
f(t)
