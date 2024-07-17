import tensorflow as tf
from tf2_test_calls3 import g


class C(tf.keras.Model):

    def call(self, a):
        assert isinstance(a, tf.Tensor)


g(C())
