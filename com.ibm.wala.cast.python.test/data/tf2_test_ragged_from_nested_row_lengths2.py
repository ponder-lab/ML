import tensorflow as tf


def test_ragged_from_nested_row_lengths_keyword(a):
    pass


nested_row_lengths = [
    [2, 1, 0, 2],
    [2, 0, 3, 1, 1],
]

x = [10, 20, 30, 40, 50, 60, 70]

# Pass nested_row_lengths as a keyword argument
arg = tf.RaggedTensor.from_nested_row_lengths(x, nested_row_lengths=nested_row_lengths)
assert arg.shape == (4, None, None)
assert arg.dtype == tf.int32

test_ragged_from_nested_row_lengths_keyword(arg)
