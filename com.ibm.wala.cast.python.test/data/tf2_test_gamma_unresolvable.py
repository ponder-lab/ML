import json

import tensorflow as tf


def consume(z):
    pass


# `tf.random.gamma`'s `shape` argument is unresolvable here (`json.loads`); the output
# rank rides on it, so the result floors to ⊤. Regression guard for wala/ML#611:
# `Gamma` previously threw "Cannot determine shape for mandatory shape parameter".
consume(tf.random.gamma(json.loads("[2]"), 1.0))
