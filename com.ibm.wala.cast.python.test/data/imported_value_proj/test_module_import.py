import tensorflow as tf
import common_model


def test_call():
    common_model.model(tf.ones([2, 4]))
