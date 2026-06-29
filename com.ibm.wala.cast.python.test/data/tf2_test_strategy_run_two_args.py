import tensorflow as tf

strategy = tf.distribute.MirroredStrategy()


def consume_inp(x):
    assert x.shape == (2,)
    assert x.dtype == tf.int32


def consume_tar(x):
    assert x.shape == (2,)
    assert x.dtype == tf.int32


def step_fn(inp, tar):
    consume_inp(inp)
    consume_tar(tar)
    return tar


a = tf.constant([1, 2], dtype=tf.int32)
b = tf.constant([3, 4], dtype=tf.int32)
strategy.run(step_fn, (a, b))
