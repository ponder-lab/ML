"""Coverage companion to `tf2_test_reshape_self_arith.py` for wala/ML#581: exercises the
`ADD` operator and a literal operand in the shape-argument arithmetic fold (the sibling fixture
covers `MUL` over two instance-field reads). `self.base + 4` mixes a field read (resolved via the
points-to analysis) with a pure literal (resolved via the symbol table).
"""

import tensorflow as tf


def consume(z):
    pass


class Reshaper:
    def __init__(self):
        self.base = 60

    def reshape_it(self, x):
        y = tf.reshape(x, [-1, self.base + 4])
        assert y.shape == (2, 64) and y.dtype == tf.float32
        consume(y)


r = Reshaper()
x = tf.ones([8, 16])
assert x.shape == (8, 16)
r.reshape_it(x)
