# A tensor-input read whose points-to set is exactly the None constant (wala/ML#961): the arm
# `if past is not None:` is dead when `past` is None, so `tf.unstack(past)` and the concat over its
# pieces cannot execute and contribute nothing; a driver passing a real tensor keeps its members.
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


def consume_with_past(h):
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

past = tf.ones((2, 2, 5, 4))
w = Block()(tf.ones((2, 3, 4)), past=past)
assert w.shape == (2, 8, 4) and w.dtype == tf.float32
consume_with_past(w)


class BlockConcat(tf.keras.layers.Layer):
    def __init__(self):
        super(BlockConcat, self).__init__()
        self.dense = tf.keras.layers.Dense(4)

    def call(self, x, past=None):
        h = self.dense(x)
        if past is not None:
            h = tf.concat([past, h], axis=-2)
        return h


def consume_direct_concat(h):
    pass


c = BlockConcat()(tf.ones((2, 3, 4)), past=None)
assert c.shape == (2, 3, 4) and c.dtype == tf.float32
consume_direct_concat(c)


def consume_list_element_control(m):
    pass


class BlockListElement(tf.keras.layers.Layer):
    def __init__(self):
        super(BlockListElement, self).__init__()
        self.dense = tf.keras.layers.Dense(4)

    def call(self, x):
        h = self.dense(x)
        # A real list element beside the tensor: `tf.concat` accepts it (converted), so the
        # result is a tensor whatever the analysis knows about `h`.
        return tf.concat([[[[1.0, 2.0, 3.0, 4.0]] * 3] * 2, h], axis=-2)


m = BlockListElement()(tf.ones((2, 3, 4)))
assert m.shape == (2, 6, 4) and m.dtype == tf.float32
consume_list_element_control(m)


def consume_nested(n):
    return n


class BlockNested(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, h, past=None):
        if past is not None:
            past_key, past_value = tf.unstack(past, axis=1)
            # The outer concat's element is itself a concat over the infeasible piece: the rule
            # follows the element to its producer and through it to the unstack.
            h = tf.concat([tf.concat([past_key, h], axis=-2), past_value], axis=-2)
        return h


n = BlockNested()(tf.ones((2, 3, 4)))
assert n.shape == (2, 3, 4) and n.dtype == tf.float32
consume_nested(n)


def consume_second(s):
    return s


class BlockSecond(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, h, past=None):
        if past is not None:
            past_key, past_value = tf.unstack(past, axis=1)
            # The infeasible piece is the second element, so the rule must read past the first.
            h = tf.concat([h, past_key], axis=-2)
        return h


s = BlockSecond()(tf.ones((2, 3, 4)))
assert s.shape == (2, 3, 4) and s.dtype == tf.float32
consume_second(s)
