import tensorflow as tf


def add(a, b):
    assert a.shape == (5, None)
    assert a.dtype == tf.int32
    assert b.shape == (5, None)
    assert b.dtype == tf.int32
    return a + b


c = add(
    tf.RaggedTensor.from_row_starts([3, 1, 4, 1, 5, 9, 2, 6], [0, 4, 4, 7, 8]),
    tf.RaggedTensor.from_row_starts([3, 11, 4, 11, 5, 19, 21, 6], [0, 4, 4, 7, 8]),
)

assert c.shape == (5, None)
assert c.dtype == tf.int32
