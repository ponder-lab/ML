import tensorflow as tf


def consume(x):
    pass


a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
c = a + b

y = tf.constant([100, 200])

ds = tf.data.Dataset.from_tensor_slices((c, y))
for x, label in ds:
    consume(x)
