import tensorflow as tf
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor


def test_keywords(rt_splits, rt_lengths, rt_rowids):
    pass


# Values: [3, 1, 4, 1, 5, 9, 2, 6] (8 items)
# Structure:
# Outer Row 0: [[3, 1], [], [4, 1, 5]] (3 inner rows)
# Outer Row 1: [[9, 2, 6]] (1 inner row)

# from_nested_row_splits
# Inner splits (8 values -> 4 rows): [0, 2, 2, 5, 8]
# Outer splits (4 rows -> 2 outer rows): [0, 3, 4]
rt_splits = RaggedTensor.from_nested_row_splits(
    flat_values=[3, 1, 4, 1, 5, 9, 2, 6], nested_row_splits=[[0, 3, 4], [0, 2, 2, 5, 8]]
)
assert rt_splits.shape.as_list() == [2, None, None]
assert rt_splits.dtype == tf.int32

# from_nested_row_lengths
# Inner lengths: [2, 0, 3, 3]
# Outer lengths: [3, 1]
rt_lengths = RaggedTensor.from_nested_row_lengths(
    flat_values=[3, 1, 4, 1, 5, 9, 2, 6], nested_row_lengths=[[3, 1], [2, 0, 3, 3]]
)
assert rt_lengths.shape.as_list() == [2, None, None]
assert rt_lengths.dtype == tf.int32

# from_nested_value_rowids
# Inner rowids (8 values): [0, 0, 2, 2, 2, 3, 3, 3]
# Outer rowids (4 inner rows): [0, 0, 0, 1]
rt_rowids = RaggedTensor.from_nested_value_rowids(
    flat_values=[3, 1, 4, 1, 5, 9, 2, 6],
    nested_value_rowids=[[0, 0, 0, 1], [0, 0, 2, 2, 2, 3, 3, 3]],
)
assert rt_rowids.shape.as_list() == [2, None, None]
assert rt_rowids.dtype == tf.int32

test_keywords(rt_splits, rt_lengths, rt_rowids)
