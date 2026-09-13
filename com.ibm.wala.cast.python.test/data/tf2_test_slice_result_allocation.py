# A slice of a tensor is a tensor of its own (wala/ML#916). A generator that reads its operand
# through the points-to set used to see the receiver's pre-slice window, because the slice result
# had no allocation of its own: an `Embedding` over `data[:, :-1]` read the receiver's 2049 where
# the slice has 2048. The two windows below are the two forms that surfaced (a `length + 1` batch
# sliced to its first `length` columns, and a `length * 2` batch sliced in half); the unsliced
# embedding is the control, and a list slice used as a shape vector is the container remainder,
# which keeps its pass-through and must not move.
import numpy as np
import tensorflow as tf

LENGTH = 2048
VOCAB = 50
EMB = 8


def consume_first_window(x):
    pass


def consume_half_window(x):
    pass


def consume_unsliced(x):
    pass


def consume_list_slice_shape(x):
    pass


def consume_ndarray_slice(x):
    pass


def consume_loop_slice(x):
    pass


embedding = tf.keras.layers.Embedding(VOCAB, EMB)

# `slide_seq2seq_batch`: a batch of length + 1 sliced to its first `length` columns.
data = tf.constant(np.random.randint(0, VOCAB, size=(2, LENGTH + 1)), dtype=tf.int32)
x = data[:, :-1]
assert x.shape == (2, LENGTH)
first = embedding(x)
assert first.shape == (2, LENGTH, EMB) and first.dtype == tf.float32
consume_first_window(first)

# `seq2seq_batch`: a batch of length * 2 sliced to its first half.
data2 = tf.constant(np.random.randint(0, VOCAB, size=(2, LENGTH * 2)), dtype=tf.int32)
y = data2[:, :LENGTH]
assert y.shape == (2, LENGTH)
half = embedding(y)
assert half.shape == (2, LENGTH, EMB)
consume_half_window(half)

# Control: the unsliced batch keeps its own extent.
whole = embedding(data)
assert whole.shape == (2, LENGTH + 1, EMB)
consume_unsliced(whole)

# Remainder: a list slice used as a shape vector is a container, not a tensor; it passes through.
shape = [4, 6, 7]
z = tf.reshape(tf.zeros((24,)), shape[:2])
assert z.shape == (4, 6)
consume_list_slice_shape(z)

# An ndarray slice keeps the pass-through (its methods are per-allocation fields in the array model,
# so a fresh allocation would lose them); it already reads its own extent through the slice pin.
arr = np.zeros((10, 6), dtype=np.float32)
row_slice = arr[:4]
assert row_slice.shape == (4, 6)
consume_ndarray_slice(tf.constant(row_slice) * 2.0)

# A loop-carried slice. This MUST be a loop, not three written-out slices: only the loop makes the
# variable depend on itself, so that the call's own result is among its receiver's objects and the
# slice generator re-enters itself when it reads that object. Three straight-line slices are a chain
# and never re-enter. The extent depends on how many times the loop ran, which the analysis does not
# fold, so the sound reading is every extent the loop could leave, from 10 down to 0, with 7 among
# them, not the once-sliced 9 alone.
loop = tf.ones((2, 10))
for _ in range(3):
    loop = loop[:, 1:]
assert loop.shape == (2, 7)
consume_loop_slice(loop)

# The builtin's two other call forms: a `slice` object made directly, and a slice of a constant.
bounds = slice(3)
assert bounds.stop == 3
prefix = "window"[:3]
assert prefix == "win"
