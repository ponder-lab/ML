import tensorflow as tf


def add(a, b):
    return a + b


nested_row_lengths = [
    tf.constant([2, 1, 0, 2], tf.int64),
    tf.constant([2, 0, 3, 1, 1], tf.int64),
]
x = tf.keras.Input(shape=[None], dtype=tf.string)
arg = tf.RaggedTensor.from_nested_row_lengths(x, nested_row_lengths)
assert arg.shape == (4, None, None, None)
assert arg.dtype == tf.string
y = add(
    arg,
    arg,
)
