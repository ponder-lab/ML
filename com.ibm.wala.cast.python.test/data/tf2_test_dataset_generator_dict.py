# Test that a dataset built by `from_generator` with a dict of `output_types` types each key of its
# element, through `padded_batch`, where a function reads the key and expands it.
import numpy as np
import tensorflow as tf


def types_only_sink(x):
    return x * 2.0


def shaped_sink(x):
    return x * 2.0


def types_only_step(batch):
    audios = batch["audios"]
    return types_only_sink(tf.expand_dims(audios, -1))


def shaped_step(batch):
    audios = batch["audios"]
    return shaped_sink(tf.expand_dims(audios, -1))


def generate():
    for length in (5, 7):
        yield {"utt_ids": b"utterance", "audios": np.ones(length, dtype=np.float32)}


types_only = tf.data.Dataset.from_generator(
    generate, output_types={"utt_ids": tf.string, "audios": tf.float32}
)
types_only = types_only.padded_batch(2, padded_shapes={"utt_ids": [], "audios": [None]})
for batch in types_only:
    result = types_only_step(batch)
    assert batch["audios"].dtype == tf.float32
    assert result.dtype == tf.float32 and result.shape == (2, 7, 1)

shaped = tf.data.Dataset.from_generator(
    generate,
    output_types={"utt_ids": tf.string, "audios": tf.float32},
    output_shapes={"utt_ids": [], "audios": [None]},
)
shaped = shaped.padded_batch(2, padded_shapes={"utt_ids": [], "audios": [None]})
for batch in shaped:
    result = shaped_step(batch)
    assert batch["audios"].dtype == tf.float32
    assert result.dtype == tf.float32 and result.shape == (2, 7, 1)


def unread_padding_sink(x):
    return x * 2.0


def generate_five():
    for _ in range(3):
        yield {"audios": np.ones(5, dtype=np.float32)}


unread = tf.data.Dataset.from_generator(
    generate_five, output_types={"audios": tf.float32}, output_shapes={"audios": [5]}
)
# The padded width is a concatenation, which the shape reader does not resolve: the padded batch is
# seven wide, not the five the generator declares.
unread = unread.padded_batch(2, padded_shapes={"audios": [7] + []})
for batch in unread:
    unread_padding_sink(batch["audios"])
    assert batch["audios"].shape[1] == 7


def signature_sink(x):
    return x * 2.0


signed = tf.data.Dataset.from_generator(
    generate,
    output_signature={
        "utt_ids": tf.TensorSpec(shape=[], dtype=tf.string),
        "audios": tf.TensorSpec(shape=[None], dtype=tf.float32),
    },
)
signed = signed.padded_batch(2, padded_shapes={"utt_ids": [], "audios": [None]})
for batch in signed:
    result = signature_sink(tf.expand_dims(batch["audios"], -1))
    assert result.dtype == tf.float32 and result.shape == (2, 7, 1)
