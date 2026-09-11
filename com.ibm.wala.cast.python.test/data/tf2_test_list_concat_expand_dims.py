"""Witness for wala/ML#907: `tf.expand_dims` over the list concatenation `[bos] + <opaque list>`.

Mirrors gpt-2-tensorflow2.0's `SequenceGenerator.sample_sequence` and its `sequence_generator.py`
driver: `bos` is a defaulted parameter the driver never passes (the driver forwards its other
options by keyword), `self.sp` is an object from an unmodeled library assigned outside `__init__`,
and its method result is the opaque right operand. Python list `+` makes the concatenation a
rank-1 list whatever the opaque operand holds (a nested operand would make the tensor conversion
raise), so the runtime value is rank 2, `(1, len)`: the batch axis is the constant 1 and only the
length is unknown to the analysis.
"""

import click
import json
import tensorflow as tf


def f(a):
    pass


class SequenceGenerator:
    def __init__(self, vocab_path):
        self.vocab_path = vocab_path
        self.sp = None

    def load_weights(self):
        self.sp = json.JSONDecoder()

    def sample_sequence(self, context=None, seq_len=512, bos=3, eos=4):
        if context == None:
            print("Give some context to model.................")
            return
        context = tf.expand_dims(([bos] + self.sp.decode(context)), 0)
        prev = context
        assert isinstance(prev, tf.Tensor)
        assert prev.shape == (1, 3)
        assert prev.dtype == tf.int32
        f(prev)


@click.command()
@click.option("--seq-len", type=int, default=512)
@click.option("--context", type=str, default="[1, 2]")
def seq_gen(seq_len, context):
    sg = SequenceGenerator("vocab")
    sg.load_weights()
    sg.sample_sequence(context, seq_len=seq_len)


if __name__ == "__main__":
    seq_gen()
