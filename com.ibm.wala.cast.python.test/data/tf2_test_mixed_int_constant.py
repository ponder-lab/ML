# Test https://github.com/wala/ML/issues/1252: an int that may be a constant or an int of unknown
# value, or a constant or a tensor's static shape, is no single constant, so a size computed from it
# is not folded.
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
