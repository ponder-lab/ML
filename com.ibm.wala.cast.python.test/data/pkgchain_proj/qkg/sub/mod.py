# Test https://github.com/wala/ML/issues/210.
import tensorflow as tf


def k(x):
    # Called as `qkg.sub.k(...)` after `import qkg`.
    return x + 1


class Scaler:
    def __call__(self, x):
        # Called as `qkg.sub.Scaler()(...)` after `import qkg`.
        return x * 2
