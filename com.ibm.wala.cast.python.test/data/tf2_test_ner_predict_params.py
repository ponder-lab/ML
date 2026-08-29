import tensorflow as tf

# The token-classification driver shape in miniature: a checkpoint-style parameter object whose
# fields the driver overrides with literals, a loader that batches by the stored size with
# `drop_remainder=True`, and a model whose `predict` receives a LIST of the three token tensors.
# Static-analysis-only (no real tfrecord at run time), following the loader-fixture precedent.


class Param:
    pass


def load_param():
    p = Param()
    p.batch_size = 32
    p.maxlen = 128
    return p


param = load_param()
param.batch_size = 8
param.maxlen = 100


def consume_x(x):
    pass


def consume_element(e):
    pass


class TFLoader:
    def __init__(self, maxlen, batch_size):
        self.maxlen = maxlen
        self.batch_size = batch_size

    def decode_record(self, record):
        feature_description = {
            "input_ids": tf.io.FixedLenFeature([self.maxlen], tf.int64),
            "segment_ids": tf.io.FixedLenFeature([self.maxlen], tf.int64),
            "input_mask": tf.io.FixedLenFeature([self.maxlen], tf.int64),
            "label_id": tf.io.FixedLenFeature([], tf.int64),
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
        dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=True)
        return dataset

    def load_mapped(self):
        raw_dataset = tf.data.TFRecordDataset("valid.tfrecords")
        return raw_dataset.map(lambda record: self.decode_record(record))


class TokenTagger(tf.keras.Model):
    def __init__(self, param):
        super(TokenTagger, self).__init__()
        self.batch_size = param.batch_size
        self.maxlen = param.maxlen
        self.dense = tf.keras.layers.Dense(9, activation="relu")

    def call(self, inputs, is_training=True):
        return self.dense(tf.cast(inputs[0], tf.float32))

    def predict(self, inputs, is_training=False):
        output = self(inputs, is_training=is_training)
        return output


model = TokenTagger(param)
model.build(input_shape=(3, param.batch_size, param.maxlen))
ner_load = TFLoader(param.maxlen, param.batch_size)

for X, token_type_id, input_mask, Y in ner_load.load_valid():
    consume_x(X)
    predict = model.predict([X, token_type_id, input_mask])

# The whole-element read: the mapped element consumed WITHOUT destructuring. The element is a
# tuple of three (100,) int64 tensors and a scalar int64 label, so the whole-element view is the
# union of exactly those component types; a (4, 100) or (4,) member would be the tuple's arity
# walked as a tensor axis, a shape the runtime never produces.
for element in ner_load.load_mapped():
    consume_element(element)
