import tensorflow as tf
from tensorflow import RaggedTensor


def add(a, b):
    return a + b


arg1 = RaggedTensor.from_row_lengths([3, 1, 4, 1, 5, 9, 2, 6], [4, 0, 3, 1, 0])
assert arg1.shape.as_list() == [5, None]
assert arg1.dtype == tf.int32
arg2 = RaggedTensor.from_row_lengths([3, 11, 4, 11, 5, 19, 21, 6], [4, 0, 3, 1, 0])
assert arg2.shape.as_list() == [5, None]
assert arg2.dtype == tf.int32
c = add(arg1, arg2)
