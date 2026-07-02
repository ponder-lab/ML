# Probe for wala/ML#618: a tensor computed inside a `with tf.name_scope(...)` block keeps its
# type (gpt-2 wraps every stage of `Gpt2.call` in one).
import tensorflow as tf


def consume(t):
    pass


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

    def call(self, inputs, training=False):
        with tf.name_scope("stage"):
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
