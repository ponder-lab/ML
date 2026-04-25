import tensorflow as tf


def add(a, b):
    return a + b


nested_value_rowids = [
    tf.constant([0, 0, 1, 3, 3], tf.int64),
    tf.constant([0, 0, 2, 2, 2, 3, 4], tf.int64),
]
x = tf.keras.Input(shape=[None], dtype=tf.string)

arg1 = tf.RaggedTensor.from_nested_value_rowids(x, nested_value_rowids)
assert arg1.shape == (4, None, None, None)
assert arg1.dtype == tf.string

arg2 = tf.RaggedTensor.from_nested_value_rowids(x, nested_value_rowids)
assert arg2.shape == (4, None, None, None)
assert arg2.dtype == tf.string

c = add(
    arg1,
    arg2,
)
