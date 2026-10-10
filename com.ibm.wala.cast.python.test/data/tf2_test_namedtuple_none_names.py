# A `collections.namedtuple` whose field names may be `None` where the analysis reads them: a helper
# that replaces a missing `names` still gives the type its names.
import collections

import tensorflow as tf


def make(names=None):
    if names is None:
        names = ("first", "second")
    return collections.namedtuple("Pair", names)


def consume_defaulted(x):
    assert x.dtype == tf.float32
    assert x.shape == ()


Pair = make()
consume_defaulted(Pair(tf.constant(1.0), tf.constant(2)).first)
