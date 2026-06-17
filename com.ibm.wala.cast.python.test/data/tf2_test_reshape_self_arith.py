"""Scoping fixture for wala/ML#581: does shape inference resolve a `tf.reshape` whose dim is
arithmetic over instance attributes (`self.heads * self.out_features`)?

Mirrors the perf-eval corpus's pervasive pattern (NLPGNN attention, MusicTransformer/gpt-2
attention, deep_recommenders CIN). The embedded interpreter cannot fold `self.X` (it evaluates
source text and has no `self`), so resolving this dim needs the generator-side extraction that
#581 reconciles.
"""

import tensorflow as tf


def consume(z):
    pass


class Attn:
    def __init__(self):
        self.heads = 8
        self.out_features = 64

    def reshape_it(self, x):
        y = tf.reshape(x, [-1, self.heads * self.out_features])
        assert y.shape == (4, 512) and y.dtype == tf.float32
        consume(y)


a = Attn()
x = tf.ones([4, 16, 32])
assert x.shape == (4, 16, 32)
a.reshape_it(x)
