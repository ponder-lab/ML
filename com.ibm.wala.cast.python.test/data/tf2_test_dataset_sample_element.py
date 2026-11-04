# Test https://github.com/wala/ML/issues/1215: the element of `sample_from_datasets` and
# `choose_from_datasets` is any one of the elements of the datasets they are given, so a tuple
# element stays a tuple whose members keep their own types.
import tensorflow as tf


def consume_sampled_pair(p):
    pass


def consume_sampled_a(a):
    pass


def consume_sampled_b(b):
    pass


def consume_sampled_batched_a(a):
    pass


def consume_chosen_pair(p):
    pass


def consume_chosen_a(a):
    pass


def consume_chosen_b(b):
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

for p in tf.data.Dataset.sample_from_datasets([pairs, others], seed=1):
    assert isinstance(p, tuple) and len(p) == 2
    consume_sampled_pair(p)

for a, b in tf.data.Dataset.sample_from_datasets([pairs, others], seed=1):
    assert a.shape in ((4,), (5,)) and a.dtype == tf.float32
    consume_sampled_a(a)
    assert b.shape in ((3,), (2,)) and b.dtype == tf.int32
    consume_sampled_b(b)

# A batch of 1, since a batch of two elements of different shapes is rejected at run time.
for a, b in tf.data.Dataset.sample_from_datasets([pairs, others], seed=1).batch(1):
    assert a.shape in ((1, 4), (1, 5)) and a.dtype == tf.float32
    consume_sampled_batched_a(a)

choice = tf.data.Dataset.range(2).repeat(4)
for p in tf.data.Dataset.choose_from_datasets([pairs, others], choice):
    assert isinstance(p, tuple) and len(p) == 2
    consume_chosen_pair(p)

for a, b in tf.data.Dataset.choose_from_datasets([pairs, others], choice):
    assert a.shape in ((4,), (5,)) and a.dtype == tf.float32
    consume_chosen_a(a)
    assert b.shape in ((3,), (2,)) and b.dtype == tf.int32
    consume_chosen_b(b)


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
model.fit(
    tf.data.Dataset.sample_from_datasets([base, base]).batch(2), epochs=1, verbose=0
)


def consume_sampled_self_batched(x):
    pass


def consume_weighted_batched(x):
    pass


def consume_appended_batched(x):
    pass


# Sizes: a sample of a dataset twice has twice its elements; a dataset whose weight is 0 yields
# none; and a list built by appending has no positions to sum.
one = tf.data.Dataset.from_tensor_slices(tf.ones([1, 4]))
last = tf.data.Dataset.from_tensor_slices(tf.ones([1, 4]))
for x in tf.data.Dataset.sample_from_datasets([one, one]).batch(2):
    assert x.shape == (2, 4)
    consume_sampled_self_batched(x)
for x in tf.data.Dataset.sample_from_datasets([one, last], weights=[1.0, 0.0]).batch(2):
    assert x.shape == (1, 4)
    consume_weighted_batched(x)
appended = [one]
appended.append(last)
for x in tf.data.Dataset.sample_from_datasets(appended).batch(2):
    assert x.shape == (2, 4)
    consume_appended_batched(x)
