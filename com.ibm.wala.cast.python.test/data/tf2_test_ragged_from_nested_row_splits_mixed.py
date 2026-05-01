import tensorflow as tf


def test_ragged_from_nested_row_splits_mixed(a):
    pass


nested_row_splits_mixed = [[0, 2, 3], [0, 2, 3, 4]]

x_mixed = [10, 20, 30, 40]

arg = tf.RaggedTensor.from_nested_row_splits(
    x_mixed, nested_row_splits=nested_row_splits_mixed
)
assert arg.shape == (2, None, None)
assert arg.dtype == tf.int32

test_ragged_from_nested_row_splits_mixed(arg)
