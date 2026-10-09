# A step function taking `*args` from `strategy.run` and forwarding its last two: the pack holds
# the two arguments the program passes, not the positions the library's summary pads its call to.
import tensorflow as tf

strategy = tf.distribute.MirroredStrategy()


def consume_last(x):
    assert x.shape == (3,)
    assert x.dtype == tf.float32


def last(first, second):
    consume_last(second)
    return second


def step_fn(*args):
    return last(*args[-2:])


a = tf.constant([1, 2], dtype=tf.int32)
b = tf.zeros((3,), dtype=tf.float32)
strategy.run(step_fn, (a, b))
