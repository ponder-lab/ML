# Test https://github.com/wala/ML/issues/993: a slice of a list or tuple literal by constant
# bounds holds only the elements in range, so an operation over the slice does not see a dropped
# element. Each collection is sliced where it arrives as a parameter.
import tensorflow as tf


def consume_list(x):
    assert x.shape == (8, 3) and x.dtype == tf.float32


def consume_tuple(x):
    assert x.shape == (8, 3) and x.dtype == tf.float32


def consume_tail(x):
    assert x.shape == (4, 1) and x.dtype == tf.int32


def consume_appended(x):
    assert x.shape == (6, 3) and x.dtype == tf.float32


def consume_expanded(x):
    assert x.shape == (4, 3, 1) and x.dtype == tf.float32


def consume_reshaped(x):
    assert x.shape == (2, 6) and x.dtype == tf.float32


def expand(x, axes):
    # A slice's collection reaches the reader of a single-element axis list.
    consume_expanded(tf.expand_dims(x, axes[0:1]))


def reshape(x, shape):
    # A slice's collection reaches the reader of a shape list.
    consume_reshaped(tf.reshape(x, shape[0:2]))


def slice_list(xs):
    consume_list(tf.concat(xs[0:-1], 0))


def slice_tuple(xs):
    consume_tuple(tf.concat(xs[:2], 0))


def slice_tail(xs):
    consume_tail(tf.concat(xs[2:], 0))


def slice_grown(xs):
    # A list that grows after it is built: `[0:-1]` of `[a, h]` plus an appended `a` is `[a, h]`.
    consume_appended(tf.concat(xs[0:-1], 0))


a = tf.ones((4, 3))
b = tf.ones((4, 3))
g = tf.zeros((4, 1), dtype=tf.int32)
h = tf.ones((2, 3))

# Each list arrives as a parameter, as a layer's list input does; the literal is built here.
slice_list([a, b, g])
slice_tuple((a, b, g))
slice_tail([a, b, g])
grown = [a, h]
grown.append(a)
slice_grown(grown)
expand(a, [-1, 0])
reshape(a, [2, 6, 1])
