# Test https://github.com/wala/ML/issues/1215: the element of `concatenate` is either dataset's
# element, so a tuple element is no tensor and each member keeps its own type.
import tensorflow as tf


def consume_pair(p):
    pass


def consume_same_a(a):
    pass


def consume_same_b(b):
    pass


def consume_other_a(a):
    pass


def consume_other_b(b):
    pass


def consume_batched_a(a):
    pass


def consume_unmodeled_a(a):
    pass


def consume_generated_a(a):
    pass


def consume_fit_inputs(inputs):
    pass


def consume_fit_x(x):
    pass


pairs = tf.data.Dataset.from_tensor_slices(
    (tf.ones([8, 4]), tf.zeros([8, 3], tf.int32))
)
others = tf.data.Dataset.from_tensor_slices(
    (tf.ones([8, 5]), tf.zeros([8, 2], tf.int32))
)

for p in pairs.concatenate(pairs):
    assert isinstance(p, tuple) and len(p) == 2
    consume_pair(p)

for a, b in pairs.concatenate(pairs):
    assert a.shape == (4,) and a.dtype == tf.float32
    consume_same_a(a)
    assert b.shape == (3,) and b.dtype == tf.int32
    consume_same_b(b)

for a, b in pairs.concatenate(others):
    assert a.shape in ((4,), (5,)) and a.dtype == tf.float32
    consume_other_a(a)
    assert b.shape in ((3,), (2,)) and b.dtype == tf.int32
    consume_other_b(b)

for a, b in pairs.concatenate(others).batch(2):
    assert a.shape in ((2, 4), (2, 5)) and a.dtype == tf.float32
    consume_batched_a(a)


def gen_pairs():
    for _ in range(4):
        yield tf.ones([4]), tf.zeros([3], tf.int32)


# A dataset whose element is not copied keeps the element its generator types.
unmodeled = tf.data.Dataset.from_generator(
    gen_pairs, output_types=(tf.float32, tf.int32)
)
for a, b in unmodeled:
    assert a.shape == (4,) and a.dtype == tf.float32
    consume_generated_a(a)

for a, b in pairs.concatenate(unmodeled):
    assert a.shape == (4,) and a.dtype == tf.float32
    consume_unmodeled_a(a)


class Net(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        consume_fit_inputs(inputs)
        x, mask = inputs
        consume_fit_x(x)
        return self.dense(x)


base = tf.data.Dataset.from_tensor_slices(
    ((tf.ones([8, 4]), tf.ones([8, 3])), tf.zeros([8]))
)
model = Net()
model.compile(optimizer="sgd", loss="mse")
model.fit(base.concatenate(base).batch(2), epochs=1, verbose=0)


def consume_self_batched(x):
    pass


def consume_picked_batched(x):
    pass


# Sizes: a dataset concatenated with itself has twice its elements, and a receiver that may be
# either of two datasets has one's size or the other's, not both.
one = tf.data.Dataset.from_tensor_slices(tf.ones([1, 4]))
two = tf.data.Dataset.from_tensor_slices(tf.ones([2, 4]))
last = tf.data.Dataset.from_tensor_slices(tf.ones([1, 4]))
for x in one.concatenate(one).batch(2):
    assert x.shape == (2, 4)
    consume_self_batched(x)
for picked in (True, False):
    for x in (one if picked else two).concatenate(last).batch(2):
        assert x.shape in ((2, 4), (1, 4))
        consume_picked_batched(x)
