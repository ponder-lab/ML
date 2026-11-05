# Test a model's `fit` over one of two maps of a `TFRecordDataset`, one returning pairs and one a
# tensor: the inputs are the tensor batched, or the pair's first member, and never the pair's label.
import os
import tempfile

import tensorflow as tf


path = os.path.join(tempfile.mkdtemp(), "records.tfrecord")
with tf.io.TFRecordWriter(path) as writer:
    for _ in range(4):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "x": tf.train.Feature(
                        float_list=tf.train.FloatList(value=[1.0] * 4)
                    ),
                    "y": tf.train.Feature(float_list=tf.train.FloatList(value=[0.0])),
                    "z": tf.train.Feature(
                        float_list=tf.train.FloatList(value=[2.0] * 3)
                    ),
                }
            )
        )
        writer.write(example.SerializeToString())


FEATURES = {
    "x": tf.io.FixedLenFeature([4], tf.float32),
    "y": tf.io.FixedLenFeature([], tf.float32),
    "z": tf.io.FixedLenFeature([3], tf.float32),
}


def parse_pair(record):
    f = tf.io.parse_single_example(record, FEATURES)
    return f["x"], f["y"]


def parse_z(record):
    f = tf.io.parse_single_example(record, FEATURES)
    return f["z"]


def consume_fit_inputs(inputs):
    pass


pairs = tf.data.TFRecordDataset([path]).map(parse_pair)
tensors = tf.data.TFRecordDataset([path]).map(parse_z)
flag = len([path]) == 2
ds = pairs if flag else tensors


class Net(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        assert inputs.shape[1:] == (3,), inputs.shape
        consume_fit_inputs(inputs)
        out = self.dense(inputs)
        self.add_loss(tf.reduce_mean(out))
        return out


model = Net()
model.compile(optimizer="sgd")
model.fit(ds.batch(2), epochs=1, verbose=0)
print("ok")
