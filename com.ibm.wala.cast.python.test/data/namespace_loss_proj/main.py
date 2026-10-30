# Test https://github.com/wala/ML/issues/1177: `custom` has no `__init__.py`, and the loss given to
# `compile` is constructed through `from custom import callback`.
import tensorflow as tf
from model import Net
from custom import callback

mt = Net()
mt.compile(optimizer="sgd", loss=callback.TransformerLoss())
mt.train_on_batch(tf.ones((4, 2)), tf.zeros((4,), tf.int64))
