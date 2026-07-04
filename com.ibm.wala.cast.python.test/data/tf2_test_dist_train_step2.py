# Probe for wala/ML#683 at subject shape (MusicTransformer-tensorflow2.0): the model base is
# `keras.Model` bound by `from tensorflow.python import keras`, and the callback args tuple has
# four elements, exceeding the two the strategy summary originally forwarded. The subject invokes
# the strategy through the deprecated `experimental_run_v2` alias; that name was removed from the
# runtime in TF 2.2, so this file calls `run`, which the summary wires to the same object.
import tensorflow as tf
from tensorflow.python import keras


class MyModel(keras.Model):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)

    def call(self, inputs, training=None):
        return inputs

    def __train_step(self, inp_tar, out_tar, lookup_mask, training):
        return inp_tar + out_tar

    def dist_train_step(self, inp_tar, out_tar, lookup_mask, training):
        return self._distribution_strategy.run(
            self.__train_step, args=(inp_tar, out_tar, lookup_mask, training)
        )


def consume(t):
    pass


strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = MyModel()

x = tf.ones((2, 3))
y = tf.ones((2, 3))
mask = tf.ones((2, 3))
out = model.dist_train_step(x, y, mask, True)
consume(out)

assert out.shape == (2, 3)
assert out.dtype == tf.float32
