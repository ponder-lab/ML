# Test a model's `fit` over a map whose function returns `(x, y)` on one call and a tensor of another
# shape on another, which is the dataset `fit` is given: the inputs are that tensor batched.
import tensorflow as tf


def consume_fit_inputs(inputs):
    pass


def parse(row, labelled):
    x = row * 2.0
    y = tf.reduce_sum(row)
    if labelled:
        return x, y
    return row[:3]


rows = tf.ones([4, 4])
labelled = tf.data.Dataset.from_tensor_slices(rows).map(lambda r: parse(r, True))
unlabelled = tf.data.Dataset.from_tensor_slices(rows).map(lambda r: parse(r, False))
for e in unlabelled.batch(2):
    assert e.shape == (2, 3)


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
