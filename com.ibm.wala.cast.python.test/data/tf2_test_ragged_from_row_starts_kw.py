import tensorflow as tf
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor


def test_kw(rt):
    return rt


# Keyword: values=[...], row_starts=[0, 4, 4, 7, 8] -> nrows=5
rt_kw = RaggedTensor.from_row_starts(
    values=[3, 1, 4, 1, 5, 9, 2, 6], row_starts=[0, 4, 4, 7, 8]
)
assert rt_kw.shape == (5, None)
assert rt_kw.dtype == tf.int32

test_kw(rt_kw)
