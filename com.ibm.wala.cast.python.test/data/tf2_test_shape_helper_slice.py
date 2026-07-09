import tensorflow as tf


def f(a):
    pass


def get_shape(t):
    return t.shape.as_list()


# Interprocedural shape-vector provenance (wala/ML#706): the shape list is
# produced by a user helper (the BERT/ALBERT `get_shape_list` pattern), so the
# def-use walk must follow the helper's return to `t.shape.as_list()` and map
# the callee parameter back to the caller argument.
t = tf.ones((4, 5, 6))
x = tf.ones((30,))
r = tf.reshape(x, get_shape(t)[-2:])
assert isinstance(r, tf.Tensor)
assert r.shape == (5, 6)
assert r.dtype == tf.float32
f(r)
