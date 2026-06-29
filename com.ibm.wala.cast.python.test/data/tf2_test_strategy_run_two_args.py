import tensorflow as tf

strategy = tf.distribute.MirroredStrategy()


def consume_inp(x):
    pass


def consume_tar(x):
    pass


def step_fn(inp, tar):
    # `inp`/`tar` are the arguments to `consume_inp`/`consume_tar`.
    assert inp.shape == (2,)
    assert inp.dtype == tf.int32
    consume_inp(inp)
    assert tar.shape == (2,)
    assert tar.dtype == tf.int32
    consume_tar(tar)
    return tar


a = tf.constant([1, 2], dtype=tf.int32)
b = tf.constant([3, 4], dtype=tf.int32)
strategy.run(step_fn, (a, b))
