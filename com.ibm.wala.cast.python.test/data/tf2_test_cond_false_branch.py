# `tf.cond` calls both of its branches, so a function reached only through `false_fn` gets a
# call-graph node and its parameters are typed (wala/ML#1029). Under `tf.function` TensorFlow
# traces both branches, so each sink's assertions run.
import tensorflow as tf


def consume_true(x):
    assert x.dtype == tf.float32
    assert x.shape == (2,)


def consume_false(x):
    assert x.dtype == tf.int32
    assert x.shape == (2,)


def on_true(a):
    consume_true(a)
    return a


def on_false(b):
    consume_false(b)
    return tf.cast(b, tf.float32)


@tf.function
def run(pred, a, b):
    return tf.cond(pred, lambda: on_true(a), lambda: on_false(b))


run(tf.constant(True), tf.ones((2,)), tf.zeros((2,), dtype=tf.int32))
