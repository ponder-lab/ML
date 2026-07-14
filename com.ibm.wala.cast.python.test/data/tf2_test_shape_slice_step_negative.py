import tensorflow as tf


def f(a):
    pass


def get_shape(t):
    return t.shape.as_list()


# Negative-step companion of tf2_test_shape_slice_step.py (wala/ML#709): a
# constant negative step reverses and strides the shape list, so the reshape
# target is computable from the resolved shape ((6, 4), every other dim of the
# reversal).
t = tf.ones((4, 5, 6))
x = tf.ones((24,))
r = tf.reshape(x, get_shape(t)[::-2])
assert r.shape == (6, 4)
assert r.dtype == tf.float32
f(r)
