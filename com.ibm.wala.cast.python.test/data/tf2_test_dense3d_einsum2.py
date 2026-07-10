import tensorflow as tf


def consume(t):
    pass


# Operand-order companion of `tf2_test_dense3d_einsum.py` (wala/ML#704): the
# weight (whose contracted dim is dynamic) comes FIRST in the equation, so the
# input's statically-known occurrence of the shared label arrives second and
# must refine the earlier dynamic one.
class DenseLayer3d(tf.keras.layers.Layer):
    def __init__(self, num_attention_heads, head_size, **kwargs):
        super(DenseLayer3d, self).__init__(**kwargs)
        self.num_attention_heads = num_attention_heads
        self.head_size = head_size

    def build(self, input_shape):
        self.hidden_size = input_shape[2]
        self.w = self.add_weight(
            name="kernel",
            shape=[self.hidden_size, self.num_attention_heads * self.head_size],
            trainable=True,
        )
        self.built = True

    def call(self, input_tensor):
        w = tf.reshape(
            self.w, [self.hidden_size, self.num_attention_heads, self.head_size]
        )
        return tf.einsum("HND,BFH->BFND", w, input_tensor)


layer = DenseLayer3d(3, 5)
x = tf.ones((2, 4, 6))
out = layer(x)

assert out.shape == (2, 4, 3, 5)
assert out.dtype == tf.float32

consume(out)
