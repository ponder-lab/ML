# A function the call graph does not reach may still run. Its `tf.keras.backend.update` calls, on
# TensorFlow imported by the enclosing frame, assign variables and resize no container.


def consume(c):
    pass


def make_step():
    import tensorflow as tf

    def step(m, v, grad):
        tf.keras.backend.update(m, 0.9 * m + grad)
        tf.keras.backend.update(v, 0.999 * v + grad * grad)

    return step


make_step()
c = [1, 2]
assert len(c) == 2 and c[0] == 1
consume(c)
