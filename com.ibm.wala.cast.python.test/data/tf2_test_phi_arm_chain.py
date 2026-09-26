# Test https://github.com/wala/ML/issues/970: a decided guard prunes its dead arm even when the
# live arm makes several calls, so it spans a chain of blocks rather than one.
import tensorflow as tf


def consume_multi(y):
    assert y.shape == (2, 6)
    assert y.dtype == tf.float32


def consume_single(y):
    assert y.shape == (2, 6)
    assert y.dtype == tf.float32


def consume_loop(y):
    assert y.dtype == tf.float32


def consume_shared(y):
    assert y.shape == (3,)
    assert y.dtype == tf.int32


class Select:
    def __init__(self, mode):
        self.mode = mode

    def run(self, a, b):
        # The same value reaches the merge along two arms, one dead and one live: `mode` is "q",
        # so the `x` and `z` arms are dead and only the fall-through keeps `a`. Pruning a dead
        # arm must not cut `a`, which the live arm still carries.
        y = a
        if self.mode == "x":
            y = b
        elif self.mode == "z":
            pass
        else:
            y = y
        consume_shared(y)


class Merge:
    def __init__(self, concat):
        self.concat = concat

    def multi(self, x):
        if self.concat is True:
            y = tf.reshape(x, [-1, 6])
            y = tf.identity(y)
        else:
            y = tf.reduce_mean(x, 1)
            y = tf.identity(y)
        consume_multi(y)

    def looped(self, x, n):
        # A decided guard inside a loop: the guard's own merge prunes its dead arm, but the loop
        # header's merge is reached from below and must stay undecided, since the trip count
        # decides which value leaves the loop.
        y = tf.reduce_mean(x, 1)
        for k in range(n):
            if self.concat is True:
                y = tf.reshape(x, [-1, 6])
        consume_loop(y)

    def single(self, x):
        if self.concat is True:
            y = tf.reshape(x, [-1, 6])
        else:
            y = tf.reduce_mean(x, 1)
        consume_single(y)


m = Merge(concat=True)
m.multi(tf.ones((2, 3, 2)))
m.single(tf.ones((2, 3, 2)))
m.looped(tf.ones((2, 3, 2)), 0)
m.looped(tf.ones((2, 3, 2)), 2)
Select("q").run(tf.zeros((3,), dtype=tf.int32), tf.ones((2, 2)))
