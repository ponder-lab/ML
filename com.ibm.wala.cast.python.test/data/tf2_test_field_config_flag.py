# A flag that a constructor sets to a constant and then reads from a parsed configuration, so a
# guard on the flag may be taken although the constant alone would not take it.
import json

import tensorflow as tf


def consume_flagged(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32


def consume_unguarded(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32


class Holder:
    def __init__(self, flag=False, config_text=None):
        self.flag = flag
        if config_text is not None:
            config = json.loads(config_text)
            self.flag = config["flag"]

    def run(self, t):
        consume_unguarded(t)
        if self.flag:
            consume_flagged(t)
            return True
        return False


assert Holder(config_text='{"flag": true}').run(tf.ones([2, 3]))
