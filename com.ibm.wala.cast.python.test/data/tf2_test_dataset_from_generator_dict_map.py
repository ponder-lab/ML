import tensorflow as tf


def gen():
    yield {"utt_ids": "a", "mels": [1.0, 2.0]}


def load(items):
    assert isinstance(items, dict)
    consume_mels(items["mels"])
    consume_ids(items["utt_ids"])
    return items


def consume_mels(m):
    assert isinstance(m, tf.Tensor)
    assert m.dtype == tf.float32
    return m


def consume_ids(u):
    assert isinstance(u, tf.Tensor)
    assert u.dtype == tf.string
    return u


dataset = tf.data.Dataset.from_generator(
    gen, output_types={"utt_ids": tf.string, "mels": tf.float32}
)
dataset = dataset.map(lambda items: load(items))

for element in dataset:
    pass


def gen_signed():
    yield {"x": [1.0, 2.0], "n": 3}


def load_signed(element):
    assert isinstance(element, dict)
    consume_x(element["x"])
    return element


def consume_x(x):
    assert x.shape == (2,)
    assert x.dtype == tf.float32
    return x


signed = tf.data.Dataset.from_generator(
    gen_signed,
    output_signature={
        "x": tf.TensorSpec(shape=(2,), dtype=tf.float32),
        "n": tf.TensorSpec(shape=(), dtype=tf.int32),
    },
)
signed = signed.map(lambda element: load_signed(element))

for element in signed:
    pass


def gen_batched():
    yield {"utt_ids": "a", "mels": [1.0, 2.0]}
    yield {"utt_ids": "b", "mels": [3.0, 4.0]}


def load_batched(batch):
    assert isinstance(batch, dict)
    consume_batched_mels(batch["mels"])
    return batch


def consume_batched_mels(m):
    assert isinstance(m, tf.Tensor)
    assert m.dtype == tf.float32
    return m


batched = tf.data.Dataset.from_generator(
    gen_batched, output_types={"utt_ids": tf.string, "mels": tf.float32}
)
batched = batched.batch(2)
batched = batched.map(lambda batch: load_batched(batch))

for element in batched:
    pass


def load_padded(batch):
    assert isinstance(batch, dict)
    consume_padded_mels(batch["mels"])
    return batch


def consume_padded_mels(m):
    assert isinstance(m, tf.Tensor)
    assert m.dtype == tf.float32
    return m


padded = tf.data.Dataset.from_generator(
    gen_batched, output_types={"utt_ids": tf.string, "mels": tf.float32}
)
padded = padded.padded_batch(2, padded_shapes={"utt_ids": [], "mels": [None]})
padded = padded.map(lambda batch: load_padded(batch))

for element in padded:
    pass
