import json

import tensorflow as tf


def consume(v):
    pass


# The input tensor's shape is unresolvable (it flows from `tf.ones(json.loads(...))`),
# so `input.shape[:-1] + (k,)` can't be composed and the wala/ML#609 composer degrades
# to ⊤. The dtype stays precise (float32, from tf.ones).
vals, idx = tf.math.top_k(tf.ones(json.loads("[3, 4]")), k=2)
consume(vals)
