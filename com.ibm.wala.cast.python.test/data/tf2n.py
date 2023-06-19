import tensorflow as tf

def add(s, t):
  return s + t

@tf.function
def func():
  a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
  b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
  c = tf.matmul(a, b)
  add(tf.Tensor(c.op, 0, tf.float32), tf.Tensor(c.op, 0, tf.float32))

func()
