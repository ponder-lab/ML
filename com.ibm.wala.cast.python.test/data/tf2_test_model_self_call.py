# Probe for wala/ML#618: a model method calling `self(...)` (Keras convention: `Model.__call__`
# routes to `call`) and unpacking the tuple result (mirroring gpt-2's
# `predictions, _ = self(inputs, training=True)` inside `_train_step`).
import tensorflow as tf


def consume(t):
    pass


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

    def call(self, inputs, training=False):
        h = tf.linalg.matmul(inputs, inputs)
        present = tf.zeros((2, 2))
        return h, present

    def run(self, inputs):
        predictions, _ = self(inputs, training=True)
        consume(predictions)
        return predictions


model = MyModel()
out = model.run(tf.ones((4, 4)))
assert out.shape == (4, 4)
assert out.dtype == tf.float32
