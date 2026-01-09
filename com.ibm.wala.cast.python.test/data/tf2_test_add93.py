import tensorflow as tf


def add(a, b):
    assert a.shape.as_list() == [5, None]
    assert a.dtype == tf.int32
    assert b.shape.as_list() == [5, None]
    assert b.dtype == tf.int32
    return a + b


c = add(
    tf.RaggedTensor.from_row_splits([3, 1, 4, 1, 5, 9, 2, 6], [0, 4, 4, 7, 8, 8]),
    tf.RaggedTensor.from_row_splits([2, 3, 7, 17, 8, 19, 2, 6], [0, 4, 4, 7, 8, 8]),
)

assert isinstance(c, tf.RaggedTensor)
assert c.shape.as_list() == [5, None]
assert c.dtype == tf.int32
