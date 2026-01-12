import tensorflow as tf


def test_ragged_from_nested_row_splits_positional(a):
    pass


nested_row_splits_pos = [[0, 2, 3], [0, 2, 3, 4]]

x_pos = [10, 20, 30, 40]

arg = tf.RaggedTensor.from_nested_row_splits(x_pos, nested_row_splits_pos)
assert arg.shape == (2, None, None)
assert arg.dtype == tf.int32

test_ragged_from_nested_row_splits_positional(arg)
