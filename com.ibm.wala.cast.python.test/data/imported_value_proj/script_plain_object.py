import tensorflow as tf
from common_model import plain


def f():
    plain.m(tf.ones([2, 4]))


f()
