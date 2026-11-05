# Test a model's `fit` over a map of a `TFRecordDataset` whose function returns `(x, y)`: the inputs
# `call` gets are `x` alone, without `y`'s shape.
import os
import tempfile

import tensorflow as tf


def consume_fit_inputs(inputs):
    pass


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
                }
            )
        )
        writer.write(example.SerializeToString())


def parse(record):
    features = tf.io.parse_single_example(
        record,
        {
            "x": tf.io.FixedLenFeature([4], tf.float32),
            "y": tf.io.FixedLenFeature([], tf.float32),
        },
    )
    return features["x"], features["y"]


class Net(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        assert inputs.shape[1:] == (4,)
        consume_fit_inputs(inputs)
        return self.dense(inputs)


model = Net()
model.compile(optimizer="sgd", loss="mse")
model.fit(tf.data.TFRecordDataset([path]).map(parse).batch(2), epochs=1, verbose=0)
