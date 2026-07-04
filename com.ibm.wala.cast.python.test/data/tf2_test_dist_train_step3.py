# Probe for wala/ML#683 at the MusicTransformer-tensorflow2.0 encoder-decoder shape: the callback
# args tuple has seven elements (six tensors and a bool), the widest form the subject passes to the
# strategy. The subject invokes the strategy through the deprecated `experimental_run_v2` alias;
# that name was removed from the runtime in TF 2.2, so this file calls `run`, which the summary
# wires to the same object.
import tensorflow as tf
from tensorflow.python import keras


class MyModel(keras.Model):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)

    def call(self, inputs, training=None):
        return inputs

    def __train_step(
        self, inp, inp_tar, out_tar, enc_mask, tar_mask, lookup_mask, training
    ):
        return lookup_mask + tar_mask

    def dist_train_step(
        self, inp, inp_tar, out_tar, enc_mask, tar_mask, lookup_mask, training
    ):
        return self._distribution_strategy.run(
            self.__train_step,
            args=(inp, inp_tar, out_tar, enc_mask, tar_mask, lookup_mask, training),
        )


def consume(t):
    pass


strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = MyModel()

inp = tf.ones((2, 3))
inp_tar = tf.ones((2, 3))
out_tar = tf.ones((2, 3))
enc_mask = tf.ones((2, 3))
tar_mask = tf.ones((2, 3))
lookup_mask = tf.ones((2, 3))
out = model.dist_train_step(
    inp, inp_tar, out_tar, enc_mask, tar_mask, lookup_mask, True
)
consume(out)

assert out.shape == (2, 3)
assert out.dtype == tf.float32
