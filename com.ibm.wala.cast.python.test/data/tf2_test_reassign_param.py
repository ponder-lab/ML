import tensorflow as tf

# wala/ML#907 TOP-producer test: does reassigning a parameter to a differently-shaped value union
# that value into the PARAMETER's own type set? A concrete reshape to (40,) makes the two members
# distinguishable — {(2, 20), (40,)} would show the union, {(2, 20)} alone would clear it. The pair
# differs in exactly one thing: whether the reshape result is bound back to the parameter name.


def reassign_fn(x):
    x = tf.reshape(x, [40])
    return x


def plain_fn(x):
    return tf.reshape(x, [40])


reassign_fn(tf.ones((2, 20)))
plain_fn(tf.ones((2, 20)))
