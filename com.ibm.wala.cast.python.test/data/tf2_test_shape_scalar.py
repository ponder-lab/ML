import tensorflow as tf


def f(a):
    pass


# A scalar operand has an empty shape vector: `tf.shape` of a rank-0 tensor is `(0,)`, a
# concrete extent, distinct from the unknown-rank case in the sibling fixture (wala/ML#943).
x = tf.ones(())
assert x.shape == ()

s = tf.shape(x)
assert isinstance(s, tf.Tensor)
assert s.shape == (0,)
assert s.dtype == tf.int32

f(s)
