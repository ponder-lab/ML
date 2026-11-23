# Test https://github.com/wala/ML/issues/1324: a field that a write through a member that is no
# string constant may set to a value no points-to set holds, as a parsed one, is not decided from its
# one constant (https://github.com/wala/ML/issues/1320).
import json

import tensorflow as tf


def consume_set_flag(x):
    assert x.shape == (2, 3)


class Settings:
    def __init__(self):
        self.flag = False

    def __setitem__(self, name, value):
        setattr(self, name, value)


def run(settings):
    if settings.flag:
        consume_set_flag(tf.ones([2, 3]))


settings = Settings()
for name in ["flag"]:
    settings[name] = json.loads("true")
run(settings)
