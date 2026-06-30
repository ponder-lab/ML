import tensorflow as tf

# End-to-end symptom-A reproduction for wala/ML#655: the full NLPGNN `BilstmAttention` structure
# (a `tf.keras.Model` whose `predict` forwards to `self(inputs, training)`, whose `call` delegates to
# a user-defined child `BiLSTM` layer built from unmodeled sublayers) fed from the real `TFLoader`
# `FixedLenFeature`/`TFRecordDataset` source. `BiLSTM.call`'s `inputs` parameter should type to
# `(128,)` int64, flowing the whole chain: the parsed `input_ids` field -> `model.predict(X)` ->
# `__call__` -> `call` -> `self.bilstm(inputs, training)` -> `BiLSTM.call`. Before the
# `FixedLenFeature` fix the source was non-tensor, so `inputs` came back non-tensor. The `__call__`
# forwarding itself was never the cause (the title hypothesis); the source was.
# Static-analysis-only (no real tfrecord at runtime).


class TFLoader(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen

    def decode_record(self, record):
        feature_description = {
            "input_ids": tf.io.FixedLenFeature([128], tf.int64),
            "label_id": tf.io.FixedLenFeature([], tf.int64),
            "segment_ids": tf.io.FixedLenFeature([128], tf.int64),
            "input_mask": tf.io.FixedLenFeature([128], tf.int64),
        }
        example = tf.io.parse_single_example(record, feature_description)
        return (
            example["input_ids"],
            example["segment_ids"],
            example["input_mask"],
            example["label_id"],
        )

    def load_valid(self):
        raw_dataset = tf.data.TFRecordDataset("valid.tfrecords")
        dataset = raw_dataset.map(lambda record: self.decode_record(record))
        dataset = dataset.prefetch(1)
        return dataset


class BiLSTM(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embedding_dims, hidden_dim, **kwargs):
        super(BiLSTM, self).__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dims, input_length=maxlen
        )
        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_dim, return_sequences=True)
        )

    def call(self, inputs, training):
        embed = self.embedding(inputs)
        logits = self.bilstm(embed, training=training)
        return logits


class BilstmAttention(tf.keras.Model):
    def __init__(self, maxlen, vocab_size, embedding_dims, hidden_dim, **kwargs):
        super(BilstmAttention, self).__init__(**kwargs)
        self.bilstm = BiLSTM(maxlen, vocab_size, embedding_dims, hidden_dim)
        self.dense = tf.keras.layers.Dense(2, activation="softmax")

    def call(self, inputs, training=True):
        logits = self.bilstm(inputs, training)
        logits = self.dense(logits)
        return logits

    def predict(self, inputs, training=False):
        out = self(inputs, training)
        return out


model = BilstmAttention(128, 30522, 100, 50)
load = TFLoader(128)
for X, token_type_id, input_mask, Y in load.load_valid():
    predict = model.predict(X)
