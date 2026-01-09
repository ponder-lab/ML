import tensorflow as tf


def test_ragged_from_row_limits(rt):
    pass


values = [3, 1, 4, 1, 5, 9, 2, 6]
row_limits = [4, 4, 6, 8]
rt = tf.RaggedTensor.from_row_limits(values, row_limits)

assert isinstance(rt, tf.RaggedTensor)
assert rt.shape.as_list() == [4, None]
assert rt.dtype == tf.int32

test_ragged_from_row_limits(rt)
