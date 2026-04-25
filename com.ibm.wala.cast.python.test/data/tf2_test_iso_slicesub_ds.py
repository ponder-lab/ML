import tensorflow as tf


def consume(x):
    pass


a = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
a_sliced = a[:2, ..., tf.newaxis]

y = tf.constant([100, 200])

ds = tf.data.Dataset.from_tensor_slices((a_sliced, y))
for x, label in ds:
    consume(x)
