# Test that a key read on a component of a zipped dataset's element, where the zipped datasets'
# elements are dicts, types that dataset's component: a tuple of datasets selected by index, and a
# dict of datasets selected by key.
import tensorflow as tf

FROM_RANGE = len("range") > 3


def tuple_sink(x):
    return x


def pair_sink(x):
    return x


def named_sink(x):
    return x


def named_label_sink(x):
    return x


def batched_sink(x):
    return x


def either_sink(x):
    return x


def features():
    return tf.data.Dataset.from_tensor_slices(
        {"a": tf.ones((4, 3), dtype=tf.float32), "b": tf.zeros((4,), dtype=tf.int32)}
    )


def labels():
    return tf.data.Dataset.from_tensor_slices(
        {"c": tf.ones((4, 2), dtype=tf.int64), "d": tf.zeros((4,), dtype=tf.bool)}
    )


for element in tf.data.Dataset.zip((features(), tf.data.Dataset.range(4))):
    tuple_sink(element[0]["a"])
    assert element[0]["a"].dtype == tf.float32 and element[0]["a"].shape == (3,)

for element in tf.data.Dataset.zip((features(), labels())):
    pair_sink(element[1]["c"])
    assert element[1]["c"].dtype == tf.int64 and element[1]["c"].shape == (2,)

for element in tf.data.Dataset.zip(
    {"examples": features(), "labels": tf.data.Dataset.range(4)}
):
    named_sink(element["examples"]["a"])
    named_label_sink(element["labels"])
    assert element["examples"]["a"].dtype == tf.float32
    assert element["labels"].dtype == tf.int64 and element["labels"].shape == ()

for element in tf.data.Dataset.zip((features(), labels())).batch(2):
    batched_sink(element[0]["a"])
    assert element[0]["a"].dtype == tf.float32 and element[0]["a"].shape == (2, 3)

# A zip whose argument is either of two tuples, whose first datasets differ in dtype: the first
# component is either dataset's.
if FROM_RANGE:
    pair = (tf.data.Dataset.range(4), tf.data.Dataset.range(4))
else:
    pair = (
        tf.data.Dataset.from_tensor_slices(tf.ones((4, 2), dtype=tf.float32)),
        tf.data.Dataset.range(4),
    )
for element in tf.data.Dataset.zip(pair):
    either_sink(element[0])
    assert element[0].dtype == tf.int64
