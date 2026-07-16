import tensorflow as tf


def consume(t):
    pass


def get_shape_list(tensor):
    return tensor.shape.as_list()


# A reshape whose target is a shape-vector slice with an opaque bound
# (wala/ML#711): the analysis cannot resolve `k`, so the result's shape is
# unknown while its dtype survives — the in-vivo arrival state of the
# encoder inputs.
def opaque(t, k):
    return tf.reshape(t, get_shape_list(t)[0:k])


# NLPGNN's DenseLayer3dProj (wala/ML#704) in miniature: the input arrives
# with an unresolvable shape, but the explicit einsum equation proves the
# input is rank 4, and its N/D axes share their labels with the reshaped
# weight, whose extents are statically known.
class DenseLayer3dProj(tf.keras.layers.Layer):
    def __init__(self, num_attention_heads, head_size, hidden_size, **kwargs):
        super(DenseLayer3dProj, self).__init__(**kwargs)
        self.num_attention_heads = num_attention_heads
        self.head_size = head_size
        self.hidden_size = hidden_size

    def build(self, input_shape):
        self.w = self.add_weight(
            name="kernel",
            shape=[self.hidden_size, self.num_attention_heads * self.head_size],
            trainable=True,
        )
        self.built = True

    def call(self, input_tensor):
        w = tf.reshape(
            self.w, [self.num_attention_heads, self.head_size, self.hidden_size]
        )
        return tf.einsum("BFND,NDH->BFH", input_tensor, w)


layer = DenseLayer3dProj(3, 5, 6)
x0 = tf.ones((2, 4, 3, 5))
x = opaque(x0, x0.shape.ndims)
out = layer(x)

assert x.shape == (2, 4, 3, 5)
assert x.dtype == tf.float32
assert out.shape == (2, 4, 6)
assert out.dtype == tf.float32

consume(out)
