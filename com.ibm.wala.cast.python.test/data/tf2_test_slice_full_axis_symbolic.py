# Witness for the kind collapse in `Slice.sliceShape` found while diagnosing wala/ML#875.
#
# `y` carries a `SymbolicDim("?")` on axis 1 -- the reshape `-1` placeholder, minted the same way
# as in `tf2_test_reshape_mixed_placeholder.py`. Slicing it with `begin` 0 and `size` -1 takes the
# axis in full, so the output extent equals the input extent on every axis and each dimension
# could be carried through verbatim.
#
# The `size == -1` arm resolves an extent exactly only when the input dimension and the begin
# offset are both numeric, and otherwise emits `Dynamic` for a `Dynamic` input and `Unresolved`
# for everything else. A `Symbolic` extent is "everything else", so it is reported as a fixed size
# the analysis could not compute, which is a different claim from the one it arrived with.
import os

import tensorflow as tf


def consume(t):
    pass


n = int(os.environ.get("ARIADNE_TEST_N", "2"))

x = tf.ones((n, 3, 8))
batch = tf.shape(x)[0]

# `(Unresolved, ?, 8)`: an environment-read leading size beside the `-1` placeholder.
y = tf.reshape(x, [batch, -1, 8])

# A full-axis slice: `begin` is 0 and `size` is -1 on every axis, so this is the identity on shape.
sliced = tf.slice(y, [0, 0, 0], [-1, -1, -1])

assert sliced.shape == (2, 3, 8)
assert sliced.dtype == tf.float32

consume(sliced)
