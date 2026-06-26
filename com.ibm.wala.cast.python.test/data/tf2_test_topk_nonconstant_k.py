import json

import tensorflow as tf


def consume(v):
    pass


# `k` comes from an unmodeled source, so it isn't a resolvable integer constant.
# The wala/ML#609 composer can't compose `input.shape[:-1] + (k,)` and degrades to
# ⊤ rather than guessing. Exercises the non-constant-`k` path.
k = json.loads("2")
values, indices = tf.math.top_k(tf.constant([1.0, 3.0, 2.0, 5.0]), k=k)
consume(values)
