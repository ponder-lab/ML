# Probe distilled from a corpus training script: a tf.keras.Model subclass whose call returns a
# TUPLE of tf.reshape results, unpacked at the top-level call site and passed to a downstream
# function. Discriminates the reshape-producer axis against the (passing) layer-tuple-return shape.
import tensorflow as tf


def consume(scores, boxes):
    assert scores.shape == (4, 4)
    assert scores.dtype == tf.float32
    assert boxes.shape == (4, 4)
    assert boxes.dtype == tf.float32


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

    def call(self, x, training=False):
        scores = tf.reshape(x, [-1, 4])
        boxes = tf.reshape(x, [4, -1])
        return scores, boxes


model = MyModel()
imgs = tf.ones((4, 4))
s, b = model(imgs)
consume(s, b)
