import tensorflow as tf


def consume(x):
    pass


def split(record):
    a = tf.cast(record, tf.int32)
    b = tf.cast(record, tf.int64)
    return a, b


def get_loss(real, pred):
    assert real.shape == (4,)
    assert real.dtype == tf.int64
    consume(real)


def train_step(inputs, targets):
    get_loss(targets, inputs)


# The gpt-2 fit-side shape (wala/ML#618): a mapped dataset passed in a list, list-unpacked, iterated
# with enumerate and nested unpacking, then forwarded through an indirected callback into get_loss.
# real = targets = the second mapped tuple component = (4,) int64 (wala/ML#648 + wala/ML#506).
def fit(train_dataset):
    train_dataset, test_dataset = train_dataset
    train_fuc = train_step
    for _, (inputs, targets) in enumerate(train_dataset):
        train_fuc(inputs, targets)


ds = tf.data.Dataset.from_tensor_slices(tf.ones([3, 4])).map(split)
fit([ds, ds])
