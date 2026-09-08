# Regression guard from the wala/ML#875 diagnosis: `tf.reshape(x, [..., -1])` resolves the `-1`
# once per input member, and the target vector's other axes pass through literally.
#
# Both call sites below reshape to the same literal target, so the leading axes are identical in
# both results and come from the target vector rather than from the input. Only the trailing axis
# -- the one the `-1` stands for -- can vary, and it is folded exactly whenever the input member is
# fully numeric. The second site routes the leading extent through `tf.shape`, which the analysis
# still folds, so both members stay concrete: a symbolic `?` here would mean the fold lost an
# input it could have divided.
import tensorflow as tf


def consume(t):
    pass


def reshape_and_consume(x):
    # The `-1` is determined at runtime in both calls: 8 * 10 * 46 over 8 * 10 is exactly 46.
    p = tf.reshape(x, [8, 10, -1])
    assert p.shape == (8, 10, 46)
    consume(p)


# A directly-constructed, fully-numeric input.
reshape_and_consume(tf.ones((8, 10, 46)))

# The same runtime shape with the leading extent routed through `tf.shape`.
rows = tf.shape(tf.ones((8, 10, 46)))[0]
reshape_and_consume(tf.zeros((rows, 10, 46)))
