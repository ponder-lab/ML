# Test a self-attention layer's head split: `transpose_for_scores` receives a `Dense` projection of
# the `(batch, seq, hidden)` input, rank 3, and reshapes it to `(batch, seq, heads, head_size)`
# itself before transposing. A wrapper returns `(first,) + outputs[1:]` and an encoder loop reads
# element 0 of each layer's outputs, so the next layer's input is the rank-3 context, never the
# rank-4 attention probabilities the tuple may carry after it.
import tensorflow as tf


class Config(object):
    def __init__(self):
        self.hidden_size = 8
        self.num_attention_heads = 2
        self.attention_head_size = 4
        self.output_attentions = False


def consume(x):
    assert x.shape == (2, 5, 8) and x.dtype == tf.float32
    return x


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.num_attention_heads = config.num_attention_heads
        self.all_head_size = self.num_attention_heads * config.attention_head_size
        self.query = tf.keras.layers.Dense(self.all_head_size, name="query")
        self.key = tf.keras.layers.Dense(self.all_head_size, name="key")
        self.value = tf.keras.layers.Dense(self.all_head_size, name="value")
        self.config = config

    def transpose_for_scores(self, x, batch_size):
        consume(x)
        x = tf.reshape(
            x,
            (batch_size, -1, self.num_attention_heads, self.config.attention_head_size),
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=False):
        hidden_states, attention_mask = inputs
        batch_size = tf.shape(hidden_states)[0]
        query_layer = self.transpose_for_scores(self.query(hidden_states), batch_size)
        key_layer = self.transpose_for_scores(self.key(hidden_states), batch_size)
        value_layer = self.transpose_for_scores(self.value(hidden_states), batch_size)
        scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        probs = tf.nn.softmax(scores, axis=-1)
        context = tf.transpose(tf.matmul(probs, value_layer), perm=[0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, -1, self.all_head_size))
        return (context, probs) if self.config.output_attentions else (context,)


class Attention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.self_attention = SelfAttention(config, name="self")

    def call(self, inputs, training=False):
        input_tensor, attention_mask = inputs
        self_outputs = self.self_attention(
            [input_tensor, attention_mask], training=training
        )
        outputs = (self_outputs[0],) + self_outputs[1:]
        return outputs


class Encoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.layer = [Attention(config, name="layer_._{}".format(i)) for i in range(2)]

    def call(self, inputs, training=False):
        hidden_states, attention_mask = inputs
        for _, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(
                [hidden_states, attention_mask], training=training
            )
            hidden_states = layer_outputs[0]
        return (hidden_states,)


class AttentionTest(object):
    def test_attention(self):
        encoder = Encoder(Config())
        encoder([tf.ones((2, 5, 8)), tf.ones((2, 5))])


AttentionTest().test_attention()
