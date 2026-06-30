import numpy as np

import tensorflow as tf


def consume(rt):
    pass


# A resolvable scalar-bearing row comes first, so `containsScalars` confirms scalars before the
# depth walk reaches the `np.ndarray` row, which the walk does not handle and so trips the
# structural floor in `getMaximumDepthOfScalars`. wala/ML#612.
pylist = [[1], np.array([2, 3])]
rt = tf.ragged.constant(pylist)

# At runtime the ragged tensor is precise. The static analysis floors the shape to ⊤ (the
# `np.ndarray` row breaks rank determination) but keeps the `int32` dtype, since the leading scalar
# row lets it confirm scalars before the `np.ndarray` element (the captured gap this fixture
# guards).
assert rt.shape.as_list() == [2, None]
assert rt.dtype == tf.int32

consume(rt)
