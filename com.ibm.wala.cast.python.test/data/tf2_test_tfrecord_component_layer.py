import tensorflow as tf

# Test https://github.com/wala/ML/issues/1010: a component of a batched TFRecord element read by an
# operation (`tf.cast`, `tf.stack` over a list of components), not only by a sink's parameter. The
# operation reads the component's allocation, so the element's type must survive the route that
# starts at the allocation rather than at the loop's iterated value.
# Static-analysis-only (no real tfrecord at runtime).


def consume(x):
    pass


def consume_cast(x):
    pass


def consume_stack(x):
    pass


def consume_label(x):
    pass


def consume_split(x):
    pass


def split_inputs(inputs):
    first, second, third = tf.split(inputs, 3, 0)
    consume_split(tf.cast(tf.squeeze(first, axis=0), tf.int32))


class TFLoader(object):
    def __init__(self, maxlen, batch_size, task):
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.task = task

    def decode_record(self, record):
        if self.task.lower() == "ner":
            feature_description = {
                "input_ids": tf.io.FixedLenFeature([self.maxlen], tf.int64),
                "label_id": tf.io.FixedLenFeature([self.maxlen], tf.int64),
                "segment_ids": tf.io.FixedLenFeature([self.maxlen], tf.int64),
                "input_mask": tf.io.FixedLenFeature([self.maxlen], tf.int64),
            }
        else:
            feature_description = {
                "input_ids": tf.io.FixedLenFeature([self.maxlen], tf.int64),
                "label_id": tf.io.FixedLenFeature([], tf.int64),
                "segment_ids": tf.io.FixedLenFeature([self.maxlen], tf.int64),
                "input_mask": tf.io.FixedLenFeature([self.maxlen], tf.int64),
            }
        example = tf.io.parse_single_example(record, feature_description)
        return (
            example["input_ids"],
            example["segment_ids"],
            example["input_mask"],
            example["label_id"],
        )

    def load_train(self):
        raw_dataset = tf.data.TFRecordDataset("train.tfrecords")
        dataset = raw_dataset.map(map_func=lambda record: self.decode_record(record))
        dataset = dataset.shuffle(1000)
        dataset = dataset.repeat(2)
        dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


load = TFLoader(100, 8, "cls")
for X, token_type_id, input_mask, Y in load.load_train():
    consume(X)
    consume_cast(tf.cast(X, tf.int32))
    consume_stack(tf.stack([X, token_type_id, input_mask]))
    consume_label(tf.cast(Y, tf.int32))
    split_inputs([X, token_type_id, input_mask])
