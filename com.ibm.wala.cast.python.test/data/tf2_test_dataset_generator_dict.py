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
