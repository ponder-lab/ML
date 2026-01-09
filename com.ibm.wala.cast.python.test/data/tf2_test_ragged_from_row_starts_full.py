import tensorflow as tf
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor


def test_ragged_from_row_starts_full(rt_pos, rt_kw, rt_mixed, rt_name_val):
    return rt_pos


# Positional: values=[...], row_starts=[0, 4, 4, 7, 8] -> nrows=4
rt_pos = tf.RaggedTensor.from_row_starts([3, 1, 4, 1, 5, 9, 2, 6], [0, 4, 4, 7, 8])
assert isinstance(rt_pos, tf.RaggedTensor)
assert rt_pos.shape == (4, None)
assert rt_pos.dtype == tf.int32

# Keyword: same
rt_kw = RaggedTensor.from_row_starts(
    values=[3, 1, 4, 1, 5, 9, 2, 6], row_starts=[0, 4, 4, 7, 8]
)
assert isinstance(rt_kw, tf.RaggedTensor)
assert rt_kw.shape == (4, None)
assert rt_kw.dtype == tf.int32

# Mixed: same
rt_mixed = tf.RaggedTensor.from_row_starts(
    [3, 1, 4, 1, 5, 9, 2, 6], row_starts=[0, 4, 4, 7, 8]
)
assert isinstance(rt_mixed, tf.RaggedTensor)
assert rt_mixed.shape == (4, None)
assert rt_mixed.dtype == tf.int32

# Name/Validate: values length 3, row_starts=[0, 2, 3] -> nrows=2
rt_name_val = tf.RaggedTensor.from_row_starts(
    [1, 2, 3], [0, 2, 3], name="my_ragged", validate=False
)
assert isinstance(rt_name_val, tf.RaggedTensor)
assert rt_name_val.shape == (2, None)
assert rt_name_val.dtype == tf.int32

test_ragged_from_row_starts_full(rt_pos, rt_kw, rt_mixed, rt_name_val)
