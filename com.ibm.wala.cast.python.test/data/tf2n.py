import tensorflow as tf

def value_index(a,b):
  return a.value_index + b.value_index

g = tf.Graph()
with g.as_default():
  c = tf.constant(30.0)
op = g.get_operations()
operation = op[0]
c = value_index(tf.Tensor(operation, 0, tf.float32), tf.Tensor(operation, 0, tf.float32))
