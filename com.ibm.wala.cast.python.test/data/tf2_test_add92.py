import tensorflow as tf
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor


def add(a, b):
    return a + b


arg1 = tf.RaggedTensor.from_row_limits([3, 1, 4, 1, 5, 9, 2, 6], [4, 4, 7, 8, 8])
assert arg1.shape.as_list() == [5, None]
assert arg1.dtype == tf.int32
arg2 = RaggedTensor.from_row_limits([3, 11, 4, 11, 5, 19, 21, 6], [4, 4, 7, 8, 8])
assert arg2.shape.as_list() == [5, None]
assert arg2.dtype == tf.int32
c = add(arg1, arg2)
