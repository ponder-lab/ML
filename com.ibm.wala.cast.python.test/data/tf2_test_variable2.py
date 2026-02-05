import tensorflow as tf


def f(v):
    pass


v1 = tf.Variable([1, 2], name="v1", dtype=tf.int64)
assert isinstance(v1, tf.Variable)
assert v1.shape.as_list() == [2]
assert v1.dtype == tf.int64

f(v1)
