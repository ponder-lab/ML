import json

import tensorflow as tf


def consume(z):
    pass


# `tf.fill`'s `dims` argument comes from an unmodeled source (`json.loads`), so it
# can't be resolved statically. `tf.fill` is still a tensor, so the right signal is
# ⊤ (unknown shape), not a crash. Regression guard for wala/ML#606:
# `Fill.getDefaultShapes` previously threw `UnsupportedOperationException` here.
dims = json.loads("[2, 3]")
consume(tf.fill(dims, 5))
