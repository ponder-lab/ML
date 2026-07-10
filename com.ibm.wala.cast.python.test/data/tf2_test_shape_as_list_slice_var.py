import tensorflow as tf


def f(a):
    pass


# Shape-vector provenance (wala/ML#703, wala/ML#704): the slice bound is a
# negated local variable rather than a literal, mirroring NLPGNN's
# `einsum_via_matmul(input_tensor, w, num_inner_dims)` idiom
# (`input_shape[-num_inner_dims:]`). The negation must be constant-folded for
# the trailing sub-shape to resolve.
t = tf.ones((4, 5, 6))
x = tf.ones((30,))
k = 2
r = tf.reshape(x, t.shape.as_list()[-k:])
assert isinstance(r, tf.Tensor)
assert r.shape == (5, 6)
assert r.dtype == tf.float32
f(r)
