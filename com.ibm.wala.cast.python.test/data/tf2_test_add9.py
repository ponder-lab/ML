import tensorflow as tf
from tensorflow.python.ops.array_ops import ones


def add(a, b):
    assert a.shape.as_list() == [1, 2]
    assert b.shape.as_list() == [2, 2]
    return a + b


c = add(tf.ones([1, 2]), ones([2, 2]))  #  [[2., 2.], [2., 2.]]
assert c.shape.as_list() == [2, 2]
assert c.dtype == tf.float32
