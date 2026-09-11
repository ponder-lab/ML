"""Control for the wala/ML#911 provenance rule: the opaque operand of the list concatenation is a
method result on a user-class object, the closest fixture form to the gpt-2 sampler's sentencepiece
tokenizer. The receiver's attribute chain roots at a script allocation, not a tensor library, so the
sequence-concatenation stage proceeds and the runtime value is rank 2, `(1, len)`.
"""

import json
import tensorflow as tf


def f(a):
    pass


class Tokenizer:
    def encode_as_ids(self, text):
        return json.loads(text)


class SequenceGenerator:
    def __init__(self):
        self.sp = Tokenizer()

    def sample_sequence(self, context=None, bos=3):
        prev = tf.expand_dims(([bos] + self.sp.encode_as_ids(context)), 0)
        assert isinstance(prev, tf.Tensor)
        assert prev.shape == (1, 3)
        f(prev)


SequenceGenerator().sample_sequence("[1, 2]")
