import fire
import tensorflow as tf


def consume(t):
    assert t.shape == (2, 3)
    assert t.dtype == tf.float32


def main(rows=2):
    consume(tf.ones([rows, 3]))


if __name__ == "__main__":
    fire.Fire(main)
