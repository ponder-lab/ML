import tensorflow as tf


def consume(z):
    pass


# `tf.eye` with a `batch_shape` prepends the batch dimensions to the identity shape: a `(3, 3)`
# identity with `batch_shape=[2]` is `(2, 3, 3)`. wala/ML#591.
consume(tf.eye(3, batch_shape=[2]))
