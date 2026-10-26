# An elementwise operation between a tensor and a Python `int` keeps the tensor's shape, whether the
# `int` is a literal or of unknown value (wala/ML#1161).
import tensorflow as tf


class Box:
    def __init__(self, n):
        self.n = n


def by_constant_field(x, box):
    return x * box.n


def by_computed_field(x, box):
    return x * box.n


def by_parameter(x, n):
    return x * n


def by_call(x, xs):
    return x * len(xs)


def consume_constant_field(t):
    pass


def consume_computed_field(t):
    pass


def consume_parameter(t):
    pass


def consume_call(t):
    pass


xs = [1, 2, 3]
x = tf.ones([2])
constant_field = by_constant_field(x, Box(3))
computed_field = by_computed_field(x, Box(len(xs)))
parameter = by_parameter(x, len(xs))
call = by_call(x, xs)
for result in (constant_field, computed_field, parameter, call):
    assert result.shape == (2,) and result.dtype == tf.float32
consume_constant_field(constant_field)
consume_computed_field(computed_field)
consume_parameter(parameter)
consume_call(call)
