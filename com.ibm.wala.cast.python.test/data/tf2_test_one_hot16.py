import tensorflow as tf


def f(a):
    pass


arg1 = [0, 1, 2]
assert isinstance(arg1, list)
assert all(isinstance(x, int) for x in arg1)
assert len(arg1) == 3
assert tf.convert_to_tensor(arg1).dtype == tf.int32
assert tf.convert_to_tensor(arg1).shape == (3,)

arg2 = tf.one_hot(arg1, 2, None, None, -1, tf.float32)
assert isinstance(arg2, tf.Tensor)
assert arg2.dtype == tf.float32
assert arg2.shape == (3, 2)

f(arg2)
