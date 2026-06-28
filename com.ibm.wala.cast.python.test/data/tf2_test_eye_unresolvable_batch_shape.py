import json

import tensorflow as tf


def consume(z):
    pass


# `tf.eye`'s `batch_shape` is a tensor whose contents are unresolvable here (`json.loads`), so the
# number of leading batch dimensions is unknown and the overall rank can't be known: the result
# floors to ⊤. Regression guard for wala/ML#611: `Eye` previously threw "Batch shape argument for
# tf.eye() should be a list of dimensions." here, aborting the whole analysis.
consume(tf.eye(3, batch_shape=tf.constant(json.loads("[2]"))))
