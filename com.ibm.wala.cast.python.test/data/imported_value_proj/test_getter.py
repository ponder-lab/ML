import tensorflow as tf
from common_model import get_model


def test_call():
    get_model()(tf.ones([2, 4]))
