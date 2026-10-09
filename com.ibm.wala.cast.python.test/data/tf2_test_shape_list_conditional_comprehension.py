# Test a shape list built by a comprehension whose element is a conditional,
# `dynamic[i] if s is None else s`, as in a `shape_list` utility: an axis the static shape leaves
# undeclared reads as a scalar int32 tensor, also through tuple unpacking, and a declared one as its
# Python integer.
import tensorflow as tf


def consume_inline(t):
    assert t.dtype == tf.int32
    assert t.shape == ()


def consume_helper(t):
    assert t.dtype == tf.int32
    assert t.shape == ()


def consume_unpacked(t):
    assert t.dtype == tf.int32
    assert t.shape == ()


def consume_static(n):
    assert n == 4


def shape_list(x):
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


@tf.function(input_signature=[tf.TensorSpec([None, None, 4], tf.float32)])
def run(x):
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    inline = [dynamic[i] if s is None else s for i, s in enumerate(static)]
    consume_inline(inline[1])
    shape = shape_list(x)
    consume_helper(shape[1])
    consume_static(shape[2])
    _, max_len, dmodel = shape_list(x)
    consume_unpacked(max_len)


run(tf.ones((2, 5, 4)))
