import tensorflow as tf


def consume(x):
    pass


def guarded(x):
    if x.shape.ndims == 2:
        x = tf.expand_dims(x, axis=-1)
    return x


def run():
    y = guarded(tf.ones([8, 10]))
    assert y.shape == (8, 10, 1)
    consume(y)


run()
