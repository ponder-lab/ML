import tensorflow as tf
from common_model import batch, consume


def f():
    consume(tf.ones([batch, 4]))


f()
