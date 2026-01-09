import tensorflow as tf


def test_ragged_mixed_args_new(rt1, rt2, rt3):
    pass


# Case 1: Positional values, keyword value_rowids
rt1 = tf.RaggedTensor.from_value_rowids(
    [3, 1, 4, 1, 5, 9], value_rowids=[0, 0, 0, 0, 2, 2]
)
assert rt1.shape.as_list() == [3, None]
assert rt1.dtype == tf.int32

# Case 2: Positional values, keyword value_rowids, keyword nrows
rt2 = tf.RaggedTensor.from_value_rowids(
    [3, 1, 4, 1, 5, 9], value_rowids=[0, 0, 0, 0, 2, 2], nrows=5
)
assert rt2.shape.as_list() == [5, None]
assert rt2.dtype == tf.int32

# Case 3: Positional values, positional value_rowids, keyword nrows
rt3 = tf.RaggedTensor.from_value_rowids([3, 1, 4], [0, 0, 0], nrows=3)
assert rt3.shape.as_list() == [3, None]
assert rt3.dtype == tf.int32

test_ragged_mixed_args_new(rt1, rt2, rt3)
