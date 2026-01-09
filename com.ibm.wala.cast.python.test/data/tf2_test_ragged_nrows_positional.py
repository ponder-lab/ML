import tensorflow as tf

def test_ragged_nrows_positional(rt):
    pass

# Case: Positional values, positional value_rowids, positional nrows
rt = tf.RaggedTensor.from_value_rowids(
    [3, 1, 4],
    [0, 0, 0],
    3
)
assert rt.shape.as_list() == [3, None]
assert rt.dtype == tf.int32

test_ragged_nrows_positional(rt)
