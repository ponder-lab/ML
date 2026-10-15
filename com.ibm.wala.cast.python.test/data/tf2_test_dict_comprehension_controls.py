# Controls for https://github.com/wala/ML/issues/1071. Each comprehension iterates the `items()` of
# a dataset element, a dict whose components the dataset models, and each read is consumed by an
# operation, as in the issue.
import tensorflow as tf


def check(x, shape, dtype):
    assert x.shape == shape
    assert x.dtype == dtype


def direct_sink(x):
    check(x, (2, 3, 1), tf.float32)
    return x


def audios_sink(x):
    check(x, (1, 3, 1), tf.float32)
    return x


def ids_sink(x):
    check(x, (1, 5, 1), tf.int32)
    return x


def reversed_ids_sink(x):
    check(x, (1, 5, 1), tf.int32)
    return x


def reversed_audios_sink(x):
    check(x, (1, 3, 1), tf.float32)
    return x


def identity_sink(x):
    check(x, (2, 3, 1), tf.float32)
    return x


def filtered_sink(x):
    check(x, (1, 3, 1), tf.float32)
    return x


def key_dependent_sink(x):
    check(x, (2, 3, 1), tf.float32)
    return x


def unknown_keys_sink(x):
    check(x, (1, 3, 1), tf.float32)
    return x


def keyword_sink(x):
    check(x, (1, 3, 1), tf.float32)
    return x


def points_to_sink(x):
    check(x, (2, 3, 1), tf.float32)
    return x


def stacked_sink(x):
    check(x, (1, 2, 3, 1), tf.float32)
    return x


def cyclic_sink(x):
    check(x, (2, 3, 1), tf.float32)
    return x


def two_keys(batch):
    direct_sink(tf.expand_dims(batch["audios"], -1))
    sliced = {k: v[0:1] for k, v in batch.items()}
    audios_sink(tf.expand_dims(sliced["audios"], -1))
    ids_sink(tf.expand_dims(sliced["ids"], -1))


def two_keys_reversed(batch):
    sliced = {k: v[0:1] for k, v in batch.items()}
    reversed_ids_sink(tf.expand_dims(sliced["ids"], -1))
    reversed_audios_sink(tf.expand_dims(sliced["audios"], -1))


def identity(batch):
    copied = {k: v for k, v in batch.items()}
    identity_sink(tf.expand_dims(copied["audios"], -1))


def filtered(batch):
    kept = {k: v[0:1] for k, v in batch.items() if k != "ids"}
    filtered_sink(tf.expand_dims(kept["audios"], -1))


def key_dependent(batch):
    chosen = {k: (v if k == "audios" else v[0:1]) for k, v in batch.items()}
    key_dependent_sink(tf.expand_dims(chosen["audios"], -1))


def cyclic(batch):
    total = tf.zeros((2, 3), dtype=tf.float32)
    for _ in range(2):
        summed = {k: v + total for k, v in batch.items() if k == "audios"}
        total = summed["audios"]
    cyclic_sink(tf.expand_dims(total, -1))


def unknown_keys(names, values):
    source = dict(zip(names, values))
    sliced = {k: v[0:1] for k, v in source.items()}
    unknown_keys_sink(tf.expand_dims(sliced["audios"], -1))


def keywords(**tensors):
    sliced = {k: v[0:1] for k, v in tensors.items()}
    keyword_sink(tf.expand_dims(sliced["audios"], -1))


def points_to(batch):
    wrapped = {k: tf.identity(v) for k, v in batch.items()}
    points_to_sink(tf.expand_dims(wrapped["audios"], -1))


def stacked(batch):
    packed = {k: tf.stack([v]) for k, v in batch.items()}
    stacked_sink(tf.expand_dims(packed["audios"], -1))


records = tf.data.Dataset.from_tensor_slices(
    {
        "audios": tf.ones((4, 3), dtype=tf.float32),
        "ids": tf.zeros((4, 5), dtype=tf.int32),
    }
).batch(2)

for batch in records:
    two_keys(batch)
    two_keys_reversed(batch)
    identity(batch)
    filtered(batch)
    key_dependent(batch)
    points_to(batch)
    stacked(batch)
    cyclic(batch)

unknown_keys(["audios"], [tf.ones((2, 3), dtype=tf.float32)])
keywords(audios=tf.ones((2, 3), dtype=tf.float32))
