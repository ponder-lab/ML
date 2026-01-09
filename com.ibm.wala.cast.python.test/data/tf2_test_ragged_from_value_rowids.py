import tensorflow as tf
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor


def test_ragged_from_value_rowids(rt_pos, rt_kw, rt_mixed):
    return rt_pos


# Positional: values=[...], value_rowids=[0, 0, 0, 0, 2, 2, 2, 3] -> nrows=4 (max(rowids)+1)
rt_pos = tf.RaggedTensor.from_value_rowids(
    [3, 1, 4, 1, 5, 9, 2, 6], [0, 0, 0, 0, 2, 2, 2, 3]
)
assert isinstance(rt_pos, tf.RaggedTensor)
assert rt_pos.shape == (4, None)
assert rt_pos.dtype == tf.int32

# Keyword: same
rt_kw = RaggedTensor.from_value_rowids(
    values=[3, 1, 4, 1, 5, 9, 2, 6], value_rowids=[0, 0, 0, 0, 2, 2, 2, 3]
)
assert isinstance(rt_kw, tf.RaggedTensor)
assert rt_kw.shape == (4, None)
assert rt_kw.dtype == tf.int32

# Mixed: same
rt_mixed = tf.RaggedTensor.from_value_rowids(
    [3, 1, 4, 1, 5, 9, 2, 6], value_rowids=[0, 0, 0, 0, 2, 2, 2, 3]
)
assert isinstance(rt_mixed, tf.RaggedTensor)
assert rt_mixed.shape == (4, None)
assert rt_mixed.dtype == tf.int32

test_ragged_from_value_rowids(rt_pos, rt_kw, rt_mixed)
