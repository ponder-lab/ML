import tensorflow as tf


def f(a):
    pass


def get_shape(t):
    return t.shape.as_list()


# Combines wala/ML#706's interprocedural hop with wala/ML#704's negated
# variable bound: `get_shape(t)[-k:]` is structurally NLPGNN's
# `einsum_via_matmul` shape read (`get_shape_list(input_tensor)[-num_inner_dims:]`).
t = tf.ones((4, 5, 6))
x = tf.ones((30,))
k = 2
r = tf.reshape(x, get_shape(t)[-k:])
assert isinstance(r, tf.Tensor)
assert r.shape == (5, 6)
assert r.dtype == tf.float32
f(r)
