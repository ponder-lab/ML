import tensorflow as tf


def test_name_val(rt):
    return rt


# Name/Validate: values length 3, row_starts=[0, 2, 3] -> nrows=3
rt_name_val = tf.RaggedTensor.from_row_starts(
    [1, 2, 3], [0, 2, 3], name="my_ragged", validate=False
)
assert rt_name_val.shape == (3, None)
assert rt_name_val.dtype == tf.int32

test_name_val(rt_name_val)
