import tensorflow as tf


def test_ragged_from_row_lengths(rt):
    pass


values = [3, 1, 4, 1, 5, 9, 2, 6]
row_lengths = [4, 0, 2, 2]
rt = tf.RaggedTensor.from_row_lengths(values, row_lengths)

assert isinstance(rt, tf.RaggedTensor)
assert rt.shape.as_list() == [4, None]
assert rt.dtype == tf.int32

test_ragged_from_row_lengths(rt)
