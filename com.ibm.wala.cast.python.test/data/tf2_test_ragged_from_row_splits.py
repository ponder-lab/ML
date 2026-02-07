import tensorflow as tf
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor


def test_ragged_from_row_splits(rt_pos, rt_kw, rt_mixed):
    return rt_pos


# Positional: values=[...], row_splits=[0, 4, 4, 7, 8, 8]
# nrows = len(row_splits) - 1 = 6 - 1 = 5.
# shape should be (5, None)
values = [3, 1, 4, 1, 5, 9, 2, 6]
row_splits = [0, 4, 4, 7, 8, 8]

rt_pos = tf.RaggedTensor.from_row_splits(values, row_splits)
assert isinstance(rt_pos, tf.RaggedTensor)
assert rt_pos.shape == (5, None)
assert rt_pos.dtype == tf.int32

# Keyword
rt_kw = RaggedTensor.from_row_splits(values=values, row_splits=row_splits)
assert isinstance(rt_kw, tf.RaggedTensor)
assert rt_kw.shape == (5, None)
assert rt_kw.dtype == tf.int32

# Mixed
rt_mixed = tf.RaggedTensor.from_row_splits(values, row_splits=row_splits)
assert isinstance(rt_mixed, tf.RaggedTensor)
assert rt_mixed.shape == (5, None)
assert rt_mixed.dtype == tf.int32

test_ragged_from_row_splits(rt_pos, rt_kw, rt_mixed)
