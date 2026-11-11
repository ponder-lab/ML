import tensorflow as tf
from common_model import model


def test_call():
    model(tf.ones([2, 4]))
