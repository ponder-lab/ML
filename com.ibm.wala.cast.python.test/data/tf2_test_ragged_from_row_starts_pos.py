import tensorflow as tf


def test_pos(rt):
    return rt


# Positional: values=[...], row_starts=[0, 4, 4, 7, 8] -> nrows=5
rt_pos = tf.RaggedTensor.from_row_starts([3, 1, 4, 1, 5, 9, 2, 6], [0, 4, 4, 7, 8])
assert rt_pos.shape == (5, None)
assert rt_pos.dtype == tf.int32

test_pos(rt_pos)
