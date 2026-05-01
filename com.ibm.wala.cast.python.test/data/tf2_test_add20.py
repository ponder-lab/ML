from tensorflow.python import ops
import tensorflow as tf


def add(a, b):
    return a + b


v1 = ops.variables.Variable([1.0, 2.0])
assert v1.shape.as_list() == [2]
assert v1.dtype == tf.float32

v2 = ops.variables.Variable([2.0, 2.0])
assert v2.shape.as_list() == [2]
assert v2.dtype == tf.float32

c = add(v1, v2)
