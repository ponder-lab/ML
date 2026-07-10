import tensorflow as tf


def consume(t):
    pass


# NLPGNN's DenseLayer3d einsum path (wala/ML#704) in miniature: the weight is
# built flat in `build` from configuration fields, reshaped to rank 3 in
# `call`, and consumed by an explicit-output einsum.
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
        return tf.einsum("BFH,HND->BFND", input_tensor, w)


layer = DenseLayer3d(3, 5)
x = tf.ones((2, 4, 6))
out = layer(x)

assert out.shape == (2, 4, 3, 5)
assert out.dtype == tf.float32

consume(out)
