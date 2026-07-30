import tensorflow as tf


def consume_col(c):
    pass


def consume_row(r):
    pass


def generator():
    for x in [[1, 2, 3], [4, 5, 6]]:
        yield {"h_r": x, "t": x}


# A dict-valued dataset whose values are padded to a batch: `h_r` is rank 2.
ds = tf.data.Dataset.from_generator(
    generator=generator, output_types={"h_r": tf.int32, "t": tf.int32}
)
ds = ds.padded_batch(2, padded_shapes={"h_r": [None], "t": [None]}, drop_remainder=True)

for data in ds:
    row = data["h_r"]
    assert row.shape == (2, 3) and row.dtype == tf.int32
    consume_row(row)

    # Indexing a column out of the dict value drops the last axis: rank 2 -> rank 1.
    col = row[:, 0]
    assert col.shape == (2,) and col.dtype == tf.int32
    consume_col(col)
