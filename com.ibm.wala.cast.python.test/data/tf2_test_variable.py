import tensorflow as tf


def f(v):
    pass


def g(v):
    pass


v1 = tf.Variable(initial_value=[1.0, 2.0], name="v1")
assert isinstance(v1, tf.Variable)
assert v1.shape.as_list() == [2]
assert v1.dtype == tf.float32

# Explicit shape/dtype
v2 = tf.Variable(initial_value=[1, 2], dtype=tf.float32, shape=[2])
assert isinstance(v2, tf.Variable)
assert v2.shape.as_list() == [2]
assert v2.dtype == tf.float32

f(v1)
g(v2)
