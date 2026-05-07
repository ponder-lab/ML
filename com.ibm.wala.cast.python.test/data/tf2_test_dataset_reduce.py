import tensorflow as tf


def f(a):
    pass


dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
result = dataset.reduce(tf.constant(0), lambda state, x: state + x)
assert isinstance(result, tf.Tensor)
assert result.shape == ()
assert result.dtype == tf.int32

f(result)
