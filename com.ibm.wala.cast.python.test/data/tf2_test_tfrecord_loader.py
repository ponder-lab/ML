import tensorflow as tf


def consume(x):
    pass


# Faithful distillation of the wala/ML#655 NLPGNN `TFLoader.load_valid` input pipeline: a
# `TFRecordDataset` mapped (via a lambda wrapping a method) by a `parse_single_example` decoder
# returning a 4-tuple of `FixedLenFeature` dict-subscripts, then `prefetch`ed, then iterated with a
# 4-way tuple unpack. `X` (the first parsed field, `input_ids`) should type to `(128,)` int64. The
# corpus also `batch`es between `map` and `prefetch`; that prepends a batch dimension (modeled by
# `DatasetBatchGenerator`) and is orthogonal to the element-typing this guards. Static-analysis-only
# (no real tfrecord at runtime).
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


load = TFLoader(128)
for X, token_type_id, input_mask, Y in load.load_valid():
    consume(X)
