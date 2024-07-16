import tensorflow as tf


class C:

    def f(self, a):
        assert isinstance(a, tf.Tensor)
        pass


C().f(tf.constant(1))
