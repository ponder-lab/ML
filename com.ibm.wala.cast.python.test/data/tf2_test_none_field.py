# A field written onto the None constant leaks as an element of any container that may be None.
# `stash(None, t)` assigns an attribute on a receiver that is None (it raises at run time, so the
# arm never executes), and `Model.call`'s `pasts` may be None in the analysis (the else arm
# assigns the defaulted `past`), so `zip`'s wildcard element read over `pasts` enumerated the
# fields of the one global None constant and handed the stashed tensor to every block as its
# `past`. The blocks' concat then carried a shape from a tensor the program never passed them.
import tensorflow as tf


def consume(x):
    return x


class Block(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, h, past=None):
        if past is not None:
            past_key, past_value = tf.unstack(past, axis=1)
            h = tf.concat([past_key, h], axis=-2)
        return h


class Model(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.blocks = [Block() for _ in range(2)]

    def call(self, x, past=None):
        pasts = [None] * 2 if past is None else past
        for block, p in zip(self.blocks, pasts):
            x = block(x, past=p)
        return x


class Holder:
    pass


def stash(holder, t):
    holder.f = t


out = Model()(tf.ones((2, 3, 4)))
assert out.shape == (2, 3, 4) and out.dtype == tf.float32
consume(out)

stash(Holder(), tf.ones((2, 2, 5, 4)))
try:
    stash(None, tf.ones((2, 2, 5, 4)))
except AttributeError:
    pass
