import numpy as np
import tensorflow as tf


def f(a):
    pass


def get_shape(t):
    return t.shape.as_list()


# np.prod over a shape-derived list (wala/ML#707): the product of the static
# trailing dimensions folds to a constant, so the reshape's target shape is
# concrete. Mirrors NLPGNN's `einsum_via_matmul`
# (`inner_dim = np.prod(input_shape[-num_inner_dims:])`).
t = tf.ones((4, 5, 6))
x = tf.ones((2, 15))
r = tf.reshape(x, [np.prod(get_shape(t)[-2:])])
assert isinstance(r, tf.Tensor)
assert r.shape == (30,)
assert r.dtype == tf.float32
f(r)
