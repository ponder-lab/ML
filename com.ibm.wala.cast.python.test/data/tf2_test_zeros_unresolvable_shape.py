import json
import tensorflow as tf


def consume(z):
    pass


s = json.loads("[2, 3]")
consume(tf.zeros(s))
