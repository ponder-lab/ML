"""Control for the wala/ML#911 provenance walk's lexical hop: the opaque operand is a call on a name a
from-import binds from a module that is not a tensor library, so the hop reaches a binding whose chain
roots at no library allocation and the sequence-concatenation stage proceeds. The runtime value is rank
2, `(1, len)`.
"""

from json import loads
import tensorflow as tf


def f(a):
    pass


def sample(context, bos=3):
    prev = tf.expand_dims(([bos] + loads(context)), 0)
    assert isinstance(prev, tf.Tensor)
    assert prev.shape == (1, 3)
    f(prev)


sample("[1, 2]")
