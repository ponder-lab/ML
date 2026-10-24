import tensorflow as tf


def consume_sum(x):
    assert x.shape == () and x.dtype == tf.float32


def consume_quotient(x):
    assert x.shape == () and x.dtype == tf.float32


def consume_comparison(x):
    assert x.shape == () and x.dtype == tf.bool


def consume_int_sum(x):
    assert x.shape == () and x.dtype == tf.int32


def consume_literal(x):
    assert x.shape == () and x.dtype == tf.float32


consume_sum(tf.constant(0.5 + 0.25))
consume_quotient(tf.constant(1 / 2))
consume_comparison(tf.constant(1 < 2))
consume_int_sum(tf.constant(1 + 1))
consume_literal(tf.constant(0.75))
