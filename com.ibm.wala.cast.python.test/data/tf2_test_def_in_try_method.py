# Test https://github.com/wala/ML/issues/1004: as tf2_test_def_in_try.py, in a method, with a
# `finally`.
import tensorflow as tf


def sink(x):
    assert x.shape == (2,)
    return x


class Model:
    def initialize(self, x):
        try:

            def inverse(y):
                return tf.math.log(tf.exp(y))

            sink(inverse(x))
        except:
            pass
        finally:
            pass


Model().initialize(tf.ones(2))
