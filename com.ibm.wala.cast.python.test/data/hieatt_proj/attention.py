import tensorflow as tf


class HieAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_size,
        attention_size,
        w_initializer=None,
        b_initializer=None,
        u_initializer=None,
        **kwargs
    ):
        super(HieAttention, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.W_initializer = tf.keras.initializers.get(w_initializer)
        self.B_initializer = tf.keras.initializers.get(b_initializer)
        self.U_initializer = tf.keras.initializers.get(u_initializer)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="W",
            shape=[self.hidden_size, self.attention_size],
            initializer=self.W_initializer,
        )
        self.B = self.add_weight(
            name="B",
            shape=[self.attention_size],
            initializer=self.B_initializer,
        )
        self.U = self.add_weight(
            name="U",
            shape=[self.attention_size],
            initializer=self.U_initializer,
        )

    def call(self, encoder_output):  # [batch,sequence_len,feats_dim]
        if self.hidden_size != encoder_output.shape[-1]:
            raise ValueError(
                "Dim of {} and {} must equal".format("hidden_size", "encode_input")
            )
        U = tf.math.tanh(
            tf.tensordot(encoder_output, self.W, axes=1) + self.B
        )  # [batch,sequence_len, attention_size]
        A = tf.tensordot(U, self.U, axes=1)  # [batch,sequence_len]
        alphas = tf.math.softmax(A)  # [batch,sequence_len]
        output = tf.math.reduce_sum(
            encoder_output * tf.expand_dims(alphas, -1), 1
        )  # [batch,sequence_len,feats_dim]
        return output, alphas

    def get_config(self):
        config = {
            "hidden_size": self.hidden_size,
            "attention_size": self.attention_size,
            "v_initializer": tf.keras.initializers.serialize(self.W_initializer),
            "w_initializer": tf.keras.initializers.serialize(self.B_initializer),
            "u_initializer": tf.keras.initializers.serialize(self.U_initializer),
        }
        base_config = super(HieAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


# Bahdanau2015
