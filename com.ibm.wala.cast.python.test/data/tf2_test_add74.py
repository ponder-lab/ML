from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
import tensorflow


def add(a, b):
    return a + b


nested_row_lengths = [
    tensorflow.constant([2, 1, 0, 2], tensorflow.int64),
    tensorflow.constant([2, 0, 3, 1, 1], tensorflow.int64),
]
x = tensorflow.keras.Input(shape=[None], dtype=tensorflow.string)
arg1 = RaggedTensor.from_nested_row_lengths(x, nested_row_lengths)
assert arg1.shape == (4, None, None, None)
assert arg1.dtype == tensorflow.string

arg2 = RaggedTensor.from_nested_row_lengths(x, nested_row_lengths)
assert arg2.shape == (4, None, None, None)
assert arg2.dtype == tensorflow.string

y = add(
    arg1,
    arg2,
)
