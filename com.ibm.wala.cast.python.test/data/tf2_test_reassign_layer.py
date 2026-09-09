import tensorflow as tf

# wala/ML#907 control, faithful frame: a parameter reassigned to a differently-shaped result inside a
# Keras Layer's call, reached through __call__ — the frame the subject's reassignment lives in. The
# parameter x must stay (2, 20): the reassignment to (40,) does not union into the parameter.


class ReassignLayer(tf.keras.layers.Layer):
    def call(self, x):
        x = tf.reshape(x, [40])
        return x


ReassignLayer()(tf.ones((2, 20)))
