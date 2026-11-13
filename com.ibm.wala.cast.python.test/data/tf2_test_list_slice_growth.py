# Test https://github.com/wala/ML/issues/1250: a slice of a list literal holds every element the
# literal had once the program may have grown the list by a route the analysis does not record
# among its positioned elements. Each list is `[a, h, a]` when it is sliced.
import operator

import tensorflow as tf


def consume_appended(x):
    assert x.shape == (6, 3) and x.dtype == tf.float32


def consume_extended(x):
    assert x.shape == (6, 3) and x.dtype == tf.float32


def consume_inserted(x):
    assert x.shape == (6, 3) and x.dtype == tf.float32


def consume_augmented(x):
    assert x.shape == (6, 3) and x.dtype == tf.float32


def consume_assigned(x):
    assert x.shape == (6, 3) and x.dtype == tf.float32


def consume_operator(x):
    assert x.shape == (6, 3) and x.dtype == tf.float32


def slice_appended(xs):
    consume_appended(tf.concat(xs[0:-1], 0))


def slice_extended(xs):
    consume_extended(tf.concat(xs[0:-1], 0))


def slice_inserted(xs):
    consume_inserted(tf.concat(xs[0:-1], 0))


def slice_augmented(xs):
    consume_augmented(tf.concat(xs[0:-1], 0))


def slice_assigned(xs):
    consume_assigned(tf.concat(xs[0:-1], 0))


def slice_operator(xs):
    consume_operator(tf.concat(xs[0:-1], 0))


a = tf.ones((4, 3))
h = tf.ones((2, 3))

grown = [a, h]
grown.append(a)
slice_appended(grown)

grown = [a, h]
grown.extend([a])
slice_extended(grown)

grown = [a, h]
grown.insert(2, a)
slice_inserted(grown)

grown = [a, h]
grown += [a]
slice_augmented(grown)

grown = [a, h]
grown[2:] = [a]
slice_assigned(grown)

grown = [a, h]
operator.iadd(grown, [a])
slice_operator(grown)
