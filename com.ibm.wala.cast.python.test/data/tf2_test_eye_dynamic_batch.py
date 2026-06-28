import json

import tensorflow as tf


def consume(z):
    pass


# `batch_shape` is a 1-element list literal: its length (and so the output rank) is statically 1,
# but its element is content-dependent (`json.loads`). The rank is known, so precision is preserved
# rather than floored to ⊤: the leading batch axis is dynamic and the `(3, 3)` identity-matrix
# suffix stays exact, giving `(None, 3, 3)`. See https://github.com/wala/ML/issues/611.
consume(tf.eye(3, batch_shape=[json.loads("2")]))
