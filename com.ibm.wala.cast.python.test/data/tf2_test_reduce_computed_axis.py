# Test a reduction whose `axis` or `keepdims` argument is supplied but holds a value the analysis
# cannot read: it is not the omitted default, so the reduction must not collapse every axis or drop
# the kept one.
import os

import tensorflow as tf


def consume_axis(t):
    assert t.shape == (2,)


def consume_keep(t):
    assert t.shape == (2, 1)


def consume_default(t):
    assert t.shape == ()


x = tf.ones((2, 3))
axis = int(os.environ.get("REDUCE_AXIS", "1"))
consume_axis(tf.reduce_sum(x, axis=axis))
keep = os.environ.get("REDUCE_KEEP", "1") == "1"
consume_keep(tf.reduce_sum(x, axis=1, keepdims=keep))
consume_default(tf.reduce_sum(x))
