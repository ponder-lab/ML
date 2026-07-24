# Test the dead library-default field write (wala/ML#769): the factory defaults are overwritten
# before any read, so the iterator elements are (2, 10), never a default-times-override product.

import tensorflow as tf
from lib import make_param


def consume(x):
    pass


param = make_param()
param.batch_size = 2
param.maxlen = 10

dataset = tf.data.Dataset.from_tensor_slices(tf.ones((6, param.maxlen))).batch(
    batch_size=param.batch_size, drop_remainder=True
)
for x in dataset:
    assert x.shape == (2, 10)
    assert x.dtype == tf.float32
    consume(x)
