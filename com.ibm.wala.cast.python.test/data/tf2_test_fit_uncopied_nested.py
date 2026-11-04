import os
import tempfile

import tensorflow as tf


def consume_first_input(a):
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
                    "z": tf.train.Feature(
                        float_list=tf.train.FloatList(value=[0.0] * 2)
                    ),
                }
            )
        )
        writer.write(example.SerializeToString())


def split(record):
    f = tf.io.parse_single_example(
        record,
        {
            "x": tf.io.FixedLenFeature([4], tf.float32),
            "z": tf.io.FixedLenFeature([2], tf.float32),
        },
    )
    return f["x"], f["z"]


features = tf.data.TFRecordDataset([path]).map(split)
labels = tf.data.Dataset.from_tensor_slices(tf.zeros([4]))
data = tf.data.Dataset.zip((features, labels)).batch(2)


class Net(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        a, b = inputs
        assert a.shape[1:] == (4,), a.shape
        assert b.shape[1:] == (2,), b.shape
        consume_first_input(a)
        return self.dense(a) + tf.reduce_sum(b, axis=-1, keepdims=True)


model = Net()
model.compile(optimizer="sgd", loss="mse")
model.fit(data, epochs=1, verbose=0)
print("ok")
