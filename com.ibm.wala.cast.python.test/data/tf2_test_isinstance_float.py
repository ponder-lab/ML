import tensorflow as tf


def consume_guard(x):
    pass


def consume_float_arm(x):
    pass


def f(x):
    if isinstance(x, float):
        consume_float_arm(x)
    else:
        consume_guard(x)


f(tf.ones([3]))
f(2.5)


def g(x: float) -> bool:
    return x > 0.0


assert g(1.5)
assert g(tf.constant(1.5))
