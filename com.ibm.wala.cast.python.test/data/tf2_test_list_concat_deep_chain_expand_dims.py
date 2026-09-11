"""Control for the wala/ML#911 provenance walk's bound: the opaque operand is a method result reached
through an attribute chain deeper than the walk follows, on a self-referential user object, so the walk
exhausts its budget without finding a library root and the sequence-concatenation stage proceeds. The
runtime value is rank 2, `(1, len)`.
"""

import json
import tensorflow as tf


def f(a):
    pass


class Tokenizer:
    def __init__(self):
        self.x = self

    def encode_as_ids(self, text):
        return json.loads(text)


def sample(context, bos=3):
    sp = Tokenizer()
    prev = tf.expand_dims(([bos] + sp.x.x.x.x.x.x.x.encode_as_ids(context)), 0)
    assert isinstance(prev, tf.Tensor)
    assert prev.shape == (1, 3)
    f(prev)


sample("[1, 2]")
