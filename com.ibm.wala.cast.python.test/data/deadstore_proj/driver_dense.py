# The Dense-units face of wala/ML#769: the factory default flows into the layer width through a
# stored attribute (`self.width = param.maxlen`; `Dense(self.width)`), so the units field unions
# the dead default with the live override without the flow-sensitive constructor-argument chase.

import tensorflow as tf
from lib import make_param


def consume(x):
    pass


param = make_param()
param.maxlen = 10
param.batch_size = 2


class Model(tf.keras.Model):
    def __init__(self, param):
        super(Model, self).__init__()
        self.width = param.maxlen
        self.dense = tf.keras.layers.Dense(self.width)

    def call(self, inputs):
        return self.dense(inputs)


model = Model(param)
y = model(tf.ones((2, 4)))
assert y.shape == (2, 10)
assert y.dtype == tf.float32
consume(y)
