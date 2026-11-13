# Test https://github.com/wala/ML/issues/1252: an int that may be a constant or an int of unknown
# value, or a constant or a tensor's static shape, is no single constant, so a size computed from it
# is not folded.
import os
import sys

import numpy as np
import tensorflow as tf


def consume_mixed(x):
    assert x.shape == (3,)


def consume_mixed_len(x):
    assert x.shape == (4,)


v = 0 if len(sys.argv) > 99 else int(len(sys.argv))
consume_mixed(tf.ones([v + 2]))

w = 0 if len(sys.argv) > 99 else len([1.0, 2.0])
consume_mixed_len(tf.ones([w + 2]))


def consume_counter(x):
    assert x.shape in {(2,), (3,), (4,)}


i = 0
while i < 3:
    consume_counter(tf.ones([i + 2]))
    i += 1


def consume_constant_or_shape(m):
    pass


t = tf.ones([2, 3])
for flag in (True, False):
    constant_or_shape = 3 if flag else t.shape
    zeros = np.zeros(constant_or_shape)
    assert zeros.shape == ((3,) if flag else (2, 3)) and zeros.dtype == np.float64
    consume_constant_or_shape(zeros)


def consume_zeros_of_mixed(m):
    pass


for flag in (True, False):
    zeros_of_mixed = np.zeros(0 if flag else int(len(sys.argv)))
    assert zeros_of_mixed.shape == ((0,) if flag else (1,))
    consume_zeros_of_mixed(zeros_of_mixed)


def consume_guarded(x):
    pass


def guard(g):
    if g == 0:
        guarded = tf.ones([2])
    else:
        guarded = tf.ones([3])
    consume_guarded(guarded)
    return guarded


for flag in (True, False):
    assert guard(0 if flag else int(len(sys.argv))).shape == ((2,) if flag else (3,))


# A supplied argument whose points-to set reads as empty is not an omitted one (wala/ML#1022).
flag_env = os.environ.get("MIXED_FLAG") is None
k = int(os.environ.get("MIXED_K", "1"))


def consume_squeeze(t):
    pass


def consume_one_hot_depth(t):
    pass


def consume_one_hot_axis(t):
    pass


def consume_boolean_mask(t):
    pass


squeezed = tf.squeeze(tf.ones([1, 1, 3]), axis=1 if flag_env else k)
assert squeezed.shape == (1, 3)
consume_squeeze(squeezed)

by_depth = tf.one_hot(tf.constant([0, 1]), 4 if flag_env else k + 3)
assert by_depth.shape == (2, 4)
consume_one_hot_depth(by_depth)

by_axis = tf.one_hot(tf.constant([0, 1]), 4, axis=0 if flag_env else k - 1)
assert by_axis.shape == (4, 2)
consume_one_hot_axis(by_axis)

masked = tf.boolean_mask(
    tf.ones([2, 3, 4]), tf.constant([True, False, True]), axis=1 if flag_env else k
)
assert masked.shape == (2, 2, 4)
consume_boolean_mask(masked)
