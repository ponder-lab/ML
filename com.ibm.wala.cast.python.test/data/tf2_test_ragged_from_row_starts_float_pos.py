import tensorflow as tf


def test_float_pos(rt):
    return rt


# Float32 (Positional): values=[...], row_starts=[0, 4, 4, 7, 8] -> nrows=5
rt_float_pos = tf.RaggedTensor.from_row_starts(
    tf.constant([3.1, 1.2, 4.3, 1.4, 5.5, 9.6, 2.7, 6.8], dtype=tf.float32),
    [0, 4, 4, 7, 8],
)
assert rt_float_pos.shape == (5, None)
assert rt_float_pos.dtype == tf.float32

test_float_pos(rt_float_pos)
