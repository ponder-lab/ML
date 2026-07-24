# Companion to driver.py (wala/ML#769): two disagreeing same-body overrides make the chase
# decline, so the analysis keeps the points-to union rather than picking either store. At
# runtime the later store wins; the union is the documented conservative degradation.

import tensorflow as tf
from lib import make_param


def consume(x):
    pass


param = make_param()
param.maxlen = 20
param.maxlen = 30

dataset = tf.data.Dataset.from_tensor_slices(tf.ones((6, param.maxlen))).batch(
    batch_size=2, drop_remainder=True
)
for x in dataset:
    assert x.shape == (2, 30)
    assert x.dtype == tf.float32
    consume(x)
