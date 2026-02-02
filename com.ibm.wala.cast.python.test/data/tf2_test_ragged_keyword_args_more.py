import tensorflow as tf
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor


def test_keywords(rt_splits, rt_lengths, rt_limits):
    pass


# from_row_splits
rt_splits = RaggedTensor.from_row_splits(
    values=[3, 1, 4, 1, 5, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8]
)
assert rt_splits.shape.as_list() == [5, None]
assert rt_splits.dtype == tf.int32

# from_row_lengths
rt_lengths = RaggedTensor.from_row_lengths(
    values=[3, 1, 4, 1, 5, 9, 2, 6], row_lengths=[4, 0, 3, 1]
)
assert rt_lengths.shape.as_list() == [4, None]
assert rt_lengths.dtype == tf.int32

# from_row_limits
rt_limits = RaggedTensor.from_row_limits(
    values=[3, 1, 4, 1, 5, 9, 2, 6], row_limits=[4, 4, 7, 8]
)
assert rt_limits.shape.as_list() == [4, None]
assert rt_limits.dtype == tf.int32

test_keywords(rt_splits, rt_lengths, rt_limits)
