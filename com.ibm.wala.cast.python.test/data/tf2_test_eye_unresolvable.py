import json

import tensorflow as tf


def consume(z):
    pass


# `tf.eye`'s `num_rows` is unresolvable here (`json.loads`). The result is still rank-2
# (square), just with dynamic dims. Regression guard for wala/ML#611: `EyeBase` previously
# threw "num_rows parameter is required" here.
consume(tf.eye(json.loads("3")))
