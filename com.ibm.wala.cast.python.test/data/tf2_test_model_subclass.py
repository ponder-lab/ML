# Test for wala/ML#662: a user subclass of `tf.keras.Model` resolves its base class in the class
# hierarchy (through the `tensorflow/keras/Model` alias shell) instead of falling back to `object`.
import tensorflow as tf


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

    def call(self, inputs):
        return inputs


model = MyModel()
assert isinstance(model, tf.keras.Model)
assert isinstance(model, tf.keras.models.Model)
