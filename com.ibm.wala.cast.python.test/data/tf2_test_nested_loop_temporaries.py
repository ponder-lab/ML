# Test for wala/ML#1106: a loop nested in another loop's body lowers to temporaries of its own, so the
# outer loop's index never reads the inner loop's elements.
import tensorflow as tf


def load(block_index):
    assert isinstance(block_index, int)
    return block_index


def load_gen(block_index):
    assert isinstance(block_index, int)
    return block_index


def consume(t):
    assert t.dtype == tf.float32 and t.shape in ((2,), (3,))
    return t


def consume_gen(t):
    assert t.dtype == tf.float32 and t.shape == (4,)
    return t


def batches(n):
    for _ in range(n):
        yield tf.ones([4])


for block_index in range(3):
    load(block_index)
    for t in [tf.ones([2]), tf.zeros([3])]:
        consume(t)


def run_gen():
    for block_index in range(3):
        load_gen(block_index)
        for batch in batches(2):
            consume_gen(batch)


run_gen()
