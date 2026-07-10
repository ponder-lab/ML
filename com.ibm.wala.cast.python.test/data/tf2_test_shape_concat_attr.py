import tensorflow as tf


def consume(t):
    pass


def get_shape_list(tensor):
    return tensor.shape.as_list()


# A configuration attribute as a literal concat element (wala/ML#712): the
# reshape target concatenates a shape-vector slice with `[self.units]`, whose
# stored value is the constructor argument.
class Proj(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(Proj, self).__init__(**kwargs)
        self.units = units

    def call(self, t):
        return tf.reshape(t, get_shape_list(t)[:-1] + [self.units])


layer = Proj(6)
x = tf.ones((2, 4, 6))
out = layer(x)

assert out.shape == (2, 4, 6)
assert out.dtype == tf.float32

consume(out)
