import tensorflow as tf
from tensorflow.python.ops.array_ops import ones


def add(a, b):
    return a + b


x = tf.ones([1, 2])
assert x.shape.as_list() == [1, 2]
y = ones([2, 2])
assert y.shape.as_list() == [2, 2]
c = add(x, y)  #  [[2., 2.], [2., 2.]]
