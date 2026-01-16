import tensorflow as tf


def test_ragged_from_nested_row_splits_keyword(a):
    pass


nested_row_splits_kw = [[0, 2, 3], [0, 2, 3, 4]]

x_kw = [10, 20, 30, 40]

arg = tf.RaggedTensor.from_nested_row_splits(
    flat_values=x_kw, nested_row_splits=nested_row_splits_kw
)
assert arg.shape == (2, None, None)
assert arg.dtype == tf.int32

test_ragged_from_nested_row_splits_keyword(arg)
