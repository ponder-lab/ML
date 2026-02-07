from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
import tensorflow as tf


def add(a, b):
    return a + b


arg1 = RaggedTensor.from_nested_row_splits(
    [3, 1, 4, 1, 5, 9, 2, 6], ([0, 3, 3, 5], [0, 4, 4, 7, 8, 8])
)
assert arg1.shape == (3, None, None)
assert arg1.dtype == tf.int32

arg2 = RaggedTensor.from_nested_row_splits(
    [13, 1, 4, 1, 15, 9, 2, 16], ([0, 3, 3, 5], [0, 4, 4, 7, 8, 8])
)
assert arg2.shape == (3, None, None)
assert arg2.dtype == tf.int32

c = add(arg1, arg2)
