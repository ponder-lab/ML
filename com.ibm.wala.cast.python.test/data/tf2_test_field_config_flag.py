# A flag that a constructor sets to a constant and then reads from a parsed configuration, so a
# guard on the flag may be taken although the constant alone would not take it.
import json

import tensorflow as tf


def consume_flagged(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32


def consume_unguarded(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32


def consume_method_checked(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32


def consume_checked(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32


def check(flag, t):
    if flag:
        consume_checked(t)
        return True
    return False


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

    def check_method(self, flag, t):
        if flag:
            consume_method_checked(t)
            return True
        return False

    def run_method_checked(self, t):
        return self.check_method(self.flag, t)

    def run_checked(self, t):
        return check(self.flag, t)


assert Holder(config_text='{"flag": true}').run_checked(tf.ones([2, 3]))
assert Holder(config_text='{"flag": true}').run_method_checked(tf.ones([2, 3]))
assert Holder(config_text='{"flag": true}').run(tf.ones([2, 3]))


def consume_mode(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32


def reset_mode():
    global MODE
    MODE = False


def configure_mode(text):
    global MODE
    MODE = json.loads(text)["mode"]


def run_mode(t):
    global MODE
    if MODE:
        consume_mode(t)
        return True
    return False


reset_mode()
configure_mode('{"mode": true}')
assert run_mode(tf.ones([2, 3]))


def consume_phi_checked(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32


def check_phi(flag, t):
    if flag:
        consume_phi_checked(t)
        return True
    return False


def pick_flag(text):
    flag = json.loads(text)["flag"] if len(text) > 2 else False
    return check_phi(flag, tf.ones([2, 3]))


assert pick_flag('{"flag": true}')
