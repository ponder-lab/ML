import json

import tensorflow as tf


def consume(z):
    pass


# `tf.random.poisson`'s `shape` argument is unresolvable here (`json.loads`); the output
# rank rides on it, so the result floors to ⊤. Regression guard for wala/ML#611:
# `Poisson` previously threw "Cannot determine shape" here.
consume(tf.random.poisson(json.loads("[2]"), 1.0))
