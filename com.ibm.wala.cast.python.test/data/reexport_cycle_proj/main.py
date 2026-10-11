import tensorflow as tf

from aaa_waiter import Waiter


def consume(x):
    assert x.dtype == tf.float32
    assert x.shape == (2, 3)


consume(Waiter()(tf.ones((2, 3))))
