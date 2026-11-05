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
                    "z": tf.train.Feature(
                        float_list=tf.train.FloatList(value=[2.0] * 3)
                    ),
                }
            )
        )
        writer.write(example.SerializeToString())


def parse(record, labelled):
    f = tf.io.parse_single_example(
        record,
        {
            "x": tf.io.FixedLenFeature([4], tf.float32),
            "y": tf.io.FixedLenFeature([], tf.float32),
            "z": tf.io.FixedLenFeature([3], tf.float32),
        },
    )
    if labelled:
        return f["x"], f["y"]
    return f["z"] * 2.0


labelled = tf.data.TFRecordDataset([path]).map(lambda r: parse(r, True))
unlabelled = tf.data.TFRecordDataset([path]).map(lambda r: parse(r, False))


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
model.fit(unlabelled.batch(2), epochs=1, verbose=0)
print("ok")
