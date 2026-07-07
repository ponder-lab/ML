# Probe for the generator-fed model-forward shape: identical to
# tf2_test_model_tuple_reshape_return.py except the model input arrives via next() on a generator
# function, tuple-unpacked at the call site. Discriminates whether an untyped generator-supplied
# input erases the typing of the model's returned tensors.
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


def make_batches():
    while True:
        yield tf.ones((4, 4)), tf.zeros((4,))


model = MyModel()
batches = make_batches()
imgs, labels = next(batches)
s, b = model(imgs)
consume(s, b)
