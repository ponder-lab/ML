# A batched element of a dataset whose size the analysis cannot know has two batch extents: the
# declared batch and the final, shorter batch. A matmul of the element with itself pairs each extent
# with each, and the pair of the declared extent with the shorter one is still one of this value's
# two batches, not a third extent. The residual carried through the loop reads the same two.
import tensorflow as tf


def consume(t):
    pass


def consume_residual(t):
    pass


def rows():
    yield tf.ones((3, 3), dtype=tf.float32)
    yield tf.ones((3, 3), dtype=tf.float32)
    yield tf.ones((3, 3), dtype=tf.float32)


def attend(x):
    return tf.matmul(x, x, transpose_b=True)


dataset = tf.data.Dataset.from_generator(
    rows, output_types=tf.float32, output_shapes=[3, 3]
)
dataset = dataset.padded_batch(2, padded_shapes=[-1, -1])

for x in dataset:
    assert x.shape.rank == 3, x.shape
    assert x.shape[1:] == (3, 3), x.shape
    assert x.shape[0] in (1, 2), x.shape
    out = attend(x)
    assert out.shape == x.shape, out.shape
    assert out.dtype == tf.float32
    consume(out)

    h = x
    for _ in range(2):
        h = h + attend(h)
    assert h.shape == x.shape, h.shape
    assert h.dtype == tf.float32
    consume_residual(h)
