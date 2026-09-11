# The corpus form of the top_k rank loss: `k` is a TENSOR (a runtime value derived from a dynamic
# axis, tf.minimum(scalar, tf.shape(x)[1])), so it is not a compile-time constant. top_k's output is
# input.shape[:-1] + (k,); the rank is known from the input, only the last axis is unknown. Because
# k is a tensor, TensorFlow's static shape reports None for that axis, so it is Dynamic.
import tensorflow as tf


def consume_values(v):
    pass


def consume_indices(v):
    pass


x = tf.constant([[1.0, 3.0, 2.0, 5.0, 4.0], [5.0, 4.0, 3.0, 2.0, 1.0]])  # (2, 5)
k = tf.minimum(3, tf.shape(x)[1])  # a tensor, so not a compile-time constant

values, indices = tf.nn.top_k(x, k=k, sorted=False)

# Runtime shape is concrete (2, 3); the static last axis is Dynamic because k is a tensor.
assert values.shape == (2, 3)
consume_values(values)

assert indices.shape == (2, 3)
consume_indices(indices)
