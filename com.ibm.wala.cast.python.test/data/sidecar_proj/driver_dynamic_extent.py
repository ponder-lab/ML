# Fixture for wala/ML#888: an annotation supplying a concrete extent for an axis inference holds
# as `Dynamic`.
#
# `tf.keras.Input(shape=(4,))` is rank 2 with a feed-dependent leading axis, which the analysis
# records as `Dynamic` because TensorFlow's own static shape reports `None` there. A user who knows
# the batch size their program is run with can record it, and that annotation is strictly more
# precise than what it refines without contradicting any evidence: `Dynamic` says the size is
# feed-dependent, not that it is unknowable.
import tensorflow as tf


def consume(x):
    pass


inp = tf.keras.Input(shape=(4,))
consume(inp)
assert inp.shape[1] == 4
