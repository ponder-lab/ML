# Control for tf2_test_model_tuple_reshape_return.py: identical shape but the returned tuple's
# elements are elementwise results rather than tf.reshape results. If this types and the reshape
# variant does not, the gap is the reshape producer, not the Model/tuple/unpack transit.
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
        scores = x + 1.0
        boxes = x * 2.0
        return scores, boxes


model = MyModel()
imgs = tf.ones((4, 4))
s, b = model(imgs)
consume(s, b)
