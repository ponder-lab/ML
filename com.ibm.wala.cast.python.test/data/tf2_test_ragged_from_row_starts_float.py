import tensorflow as tf


def test_float(rt):
    return rt


# Float32: values=[...], row_starts=[0, 4, 4, 7, 8] -> nrows=5
rt_float = tf.RaggedTensor.from_row_starts(
    values=tf.constant([3.1, 1.2, 4.3, 1.4, 5.5, 9.6, 2.7, 6.8], dtype=tf.float32),
    row_starts=[0, 4, 4, 7, 8],
)
assert rt_float.shape == (5, None)
assert rt_float.dtype == tf.float32

test_float(rt_float)
