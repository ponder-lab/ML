import numpy as np

import tensorflow as tf


def consume(rt):
    pass


# The outer `pylist` is a real list literal, but its first element is an `np.ndarray` rather than a
# nested list/tuple, so the structural depth/scalar walk can't recognize it. `RaggedConstant` floors
# both the shape and the dtype to ⊤ rather than aborting with "Expected a list or tuple". wala/ML#612.
pylist = [np.array([1, 2]), [3]]
rt = tf.ragged.constant(pylist)

# At runtime the ragged tensor is precise; the static analysis floors both axes to ⊤ because the
# unhandled `np.ndarray` row precedes any confirmable scalar (the captured gap this fixture guards).
assert rt.shape.as_list() == [2, None]
assert rt.dtype == tf.int32

consume(rt)
