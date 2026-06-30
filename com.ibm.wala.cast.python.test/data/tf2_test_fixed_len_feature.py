import tensorflow as tf


def consume(x):
    pass


# Isolates wala/ML#655: a `FixedLenFeature` value, parsed by `parse_single_example` and read back
# through a dict subscript, should type as a dense tensor (shape from `dims`, dtype from `type`).
# The corpus `decode_record` returns such values; before the fix they were non-tensor.
example = tf.train.Example(
    features=tf.train.Features(
        feature={
            "input_ids": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[0] * 128)
            )
        }
    )
).SerializeToString()
feature_description = {
    "input_ids": tf.io.FixedLenFeature([128], tf.int64),
}
parsed = tf.io.parse_single_example(example, feature_description)
v = parsed["input_ids"]
assert v.shape == (128,)
assert v.dtype == tf.int64
consume(v)
