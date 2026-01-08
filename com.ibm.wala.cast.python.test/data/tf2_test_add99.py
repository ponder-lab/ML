from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
import tensorflow as tf


def add(a, b):
    return a + b


arg1 = RaggedTensor.from_value_rowids(
    values=[3, 1, 4, 1, 5, 9, 2, 6],
    value_rowids=[0, 0, 0, 0, 2, 2, 2, 3]
)
assert arg1.shape == (4, None)
assert arg1.dtype == tf.int32

arg2 = RaggedTensor.from_value_rowids(
    values=[3, 1, 14, 1, 5, 19, 2, 16],
    value_rowids=[0, 0, 0, 0, 2, 2, 2, 3]
)
assert arg2.shape == (4, None)
assert arg2.dtype == tf.int32

c = add(
    arg1,
    arg2,
)
assert c.shape == (4, None)
assert c.dtype == tf.int32
