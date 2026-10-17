# Test for wala/ML#1108: a program class that subclasses a built-in Keras layer, applied to a Keras
# symbolic tensor, gives a Keras symbolic tensor, which the functions it is passed to receive.
import tensorflow as tf


class Frozen(tf.keras.layers.BatchNormalization):

    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


class Plain(tf.keras.layers.BatchNormalization):
    pass


def after_frozen(x):
    return tf.nn.leaky_relu(x)


def after_plain(x):
    return tf.nn.leaky_relu(x)


inp = tf.keras.layers.Input([16, 16, 3])
frozen = after_frozen(Frozen()(inp))
plain = after_plain(Plain()(inp))
assert type(frozen).__name__ == "KerasTensor"
assert type(plain).__name__ == "KerasTensor"
