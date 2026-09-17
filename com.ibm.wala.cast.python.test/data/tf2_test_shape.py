import tensorflow as tf


def f(a):
    pass


def g(b):
    pass


x = tf.ones((2, 3, 4))
s = tf.shape(x)
assert isinstance(s, tf.Tensor)
assert s.shape == (3,)
assert s.dtype == tf.int32

n = s[0]
assert isinstance(n, tf.Tensor)
assert n.shape == ()
assert n.dtype == tf.int32

f(s)
g(n)
