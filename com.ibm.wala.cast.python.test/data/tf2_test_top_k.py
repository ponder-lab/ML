import tensorflow as tf


def f_values(a):
    pass


def f_indices(a):
    pass


# `tf.math.top_k(input, k)` returns a `(values, indices)` named tuple.
# `values` shares the input dtype (float32 here); `indices` is int32.
# Shape is `input.shape[:-1] + (k,)` for both — currently emitted as ⊤.
x = tf.constant([1.0, 3.0, 2.0, 5.0, 4.0])
result = tf.math.top_k(x, k=2)
values, indices = result.values, result.indices
assert isinstance(values, tf.Tensor)
assert isinstance(indices, tf.Tensor)
assert values.dtype == tf.float32
assert indices.dtype == tf.int32
f_values(values)
f_indices(indices)
