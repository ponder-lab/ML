# Test for https://github.com/wala/ML/issues/1074: a comparison whose only use is a condition.
import tensorflow as tf


def if_greater(x):
    if tf.reduce_sum(x) > 0.0:
        return None
    return x + 1.0


def if_less_equal(x):
    y = None
    if tf.reduce_sum(x) <= 0.0:
        y = x + 1.0
    return y


def while_greater(x):
    while tf.reduce_sum(x) > 0.0:
        x = x - 1.0
    return x


def assert_greater(x):
    assert tf.reduce_sum(x) > 0.0
    return x


def nested(x):
    def inner():
        if tf.reduce_sum(x) > 0.0:
            return x + 1.0
        return x - 1.0

    return inner()


def loop_body(i, x):
    if tf.reduce_sum(x) > 0.0:
        x = x + 1.0
    return i + 1, x


def in_while_loop(x):
    return tf.while_loop(lambda i, _: i < 2, loop_body, [tf.constant(0), x])


def if_int(x, n):
    if n > 0:
        return x + 1.0
    return x


def if_len(x, xs):
    if len(xs) > 0:
        return x + 1.0
    return x


def if_is_none(x):
    if x is None:
        return None
    return x + 1.0


def assert_shape(x):
    assert x.shape == (2, 3)
    return x


def assert_dtype(x):
    assert x.dtype == tf.float32
    return x


def if_dimension(x):
    if x.shape[0] == 2:
        return x + 1.0
    return x


def if_and(x):
    if tf.reduce_sum(x) > 0.0 and tf.reduce_max(x) < 2.0:
        return None
    return x + 1.0


def if_not(x):
    if not (tf.reduce_sum(x) > 0.0):
        return None
    return x + 1.0


def ternary(x):
    return x + 1.0 if tf.reduce_sum(x) > 0.0 else x - 1.0


def comprehension_filter(ys):
    return [y for y in ys if y > 0.0]


def if_chained(x):
    if 0.0 < tf.reduce_sum(x) < 100.0:
        return None
    return x + 1.0


t = tf.ones((2, 3))
if_greater(t)
if_less_equal(t)
while_greater(t)
assert_greater(t)
nested(t)
in_while_loop(t)
if_int(t, 1)
if_len(t, [1])
if_is_none(t)
assert_shape(t)
assert_dtype(t)
if_dimension(t)
if_and(t)
if_not(t)
ternary(t)
comprehension_filter([tf.constant(1.0), tf.constant(-1.0)])
if_chained(t)
