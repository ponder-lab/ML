# List repetition and concatenation as element-carrying lists (wala/ML#960): `[None] * n` and
# `[0] + sizes` each produce a list whose elements are the operands' elements. A layer with an
# optional `past` reached through both idioms, a concatenation of ints consumed as offsets, and a
# shape vector built by concatenation whose reshape must keep resolving.
import tensorflow as tf


class Block(tf.keras.layers.Layer):
    def __init__(self):
        super(Block, self).__init__()
        self.dense = tf.keras.layers.Dense(4)

    def call(self, x, past=None):
        h = self.dense(x)
        if past is not None:
            past_key, past_value = tf.unstack(past, axis=1)
            h = tf.concat([past_key, h], axis=-2)
        return h


def consume_repeated(h):
    pass


def consume_direct(h):
    pass


def consume_offsets(offsets):
    pass


def consume_reshaped(y):
    pass


blocks = [Block(), Block()]
pasts = [None] * len(blocks)
x = tf.ones((2, 3, 4))
for block, past in zip(blocks, pasts):
    x = block(x, past=past)
assert x.shape == (2, 3, 4) and x.dtype == tf.float32
consume_repeated(x)

d = Block()(tf.ones((2, 3, 4)), past=None)
assert d.shape == (2, 3, 4) and d.dtype == tf.float32
consume_direct(d)

sizes = [2, 3]
offsets = [0] + sizes
assert offsets == [0, 2, 3]
consume_offsets(tf.constant(offsets))

z = tf.ones((6, 4))
dims = [2] + [3, 4]
y = tf.reshape(z, dims)
assert y.shape == (2, 3, 4) and y.dtype == tf.float32
consume_reshaped(y)


def decline_np_repeated(a):
    pass


def decline_tf_concatenated(c):
    pass


def decline_zeros_repeated(zz):
    pass


# Length hazards: a reader deriving an extent from a synthesized list's field count would read 1,
# 2 and (2,) here where the runtime has 3, 3 and (2, 2). Each is pinned to NOT read the miscount.
import numpy as np

a = np.array([0] * 3)
assert a.shape == (3,)
decline_np_repeated(a)

c = tf.constant([1, 2] + [3])
assert c.shape == (3,) and c.dtype == tf.int32
decline_tf_concatenated(c)

zz = tf.zeros([2] * 2)
assert zz.shape == (2, 2) and zz.dtype == tf.float32
decline_zeros_repeated(zz)
