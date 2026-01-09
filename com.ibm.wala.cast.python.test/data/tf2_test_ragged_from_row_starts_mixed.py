import tensorflow as tf


def test_mixed(rt):
    return rt


# Mixed: values (pos), row_starts (kw)
rt_mixed = tf.RaggedTensor.from_row_starts(
    [3, 1, 4, 1, 5, 9, 2, 6], row_starts=[0, 4, 4, 7, 8]
)
assert rt_mixed.shape == (5, None)
assert rt_mixed.dtype == tf.int32

test_mixed(rt_mixed)
