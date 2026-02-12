import tensorflow as tf
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor


def add(a, b):
    return a + b


t1 = tf.RaggedTensor.from_row_splits([3, 1, 4, 1, 5, 9, 2, 6], [0, 4, 4, 7, 8, 8])
t2 = RaggedTensor.from_row_splits([2, 3, 7, 17, 8, 19, 2, 6], [0, 4, 4, 7, 8, 8])

assert t1.shape.as_list() == [5, None]
assert t1.dtype == tf.int32
assert t2.shape.as_list() == [5, None]
assert t2.dtype == tf.int32

c = add(t1, t2)

assert isinstance(c, tf.RaggedTensor)
assert c.shape.as_list() == [5, None]
assert c.dtype == tf.int32
