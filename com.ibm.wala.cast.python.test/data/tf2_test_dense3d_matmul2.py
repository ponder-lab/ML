import numpy as np
import tensorflow as tf


def consume(t):
    pass


def get_shape_list(tensor):
    return tensor.shape.as_list()


def einsum_via_matmul(input_tensor, w, num_inner_dims):
    assert input_tensor.shape == (2, 4, 6)
    assert w.shape == (6, 3, 5)
    input_shape = get_shape_list(input_tensor)
    w_shape = get_shape_list(w)
    batch_dims = input_shape[:-num_inner_dims]
    inner_dims = input_shape[-num_inner_dims:]
    outer_dims = w_shape[num_inner_dims:]
    inner_dim = np.prod(inner_dims)
    outer_dim = np.prod(outer_dims)
    if num_inner_dims > 1:
        input_tensor = tf.reshape(input_tensor, batch_dims + [inner_dim])
    if len(w_shape) > 2:
        w = tf.reshape(w, [inner_dim, outer_dim])
    ret = tf.matmul(input_tensor, w)
    if len(outer_dims) > 1:
        ret = tf.reshape(ret, batch_dims + outer_dims)
    return ret


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
        return einsum_via_matmul(input_tensor, w, 1)


# NLPGNN's attention dispatch (wala/ML#704) in miniature: the DenseLayer3d is
# created in the *outer* layer's `build` (lazy), stored as an attribute, and
# invoked through it in the outer `call` — two trampoline hops before
# `einsum_via_matmul` sees the input tensor.
class Attention(tf.keras.layers.Layer):
    def __init__(self, num_attention_heads, head_size, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.num_attention_heads = num_attention_heads
        self.head_size = head_size

    def build(self, input_shape):
        self.q = DenseLayer3d(self.num_attention_heads, self.head_size)
        self.built = True

    def call(self, from_tensor):
        return self.q(from_tensor)


layer = Attention(3, 5)
x = tf.ones((2, 4, 6))
out = layer(x)

assert out.shape == (2, 4, 3, 5)
assert out.dtype == tf.float32

consume(out)
