import tensorflow as tf


def linear(input_, output_size, name, stddev=0.02, bias_start=0.0,
           reuse=tf.AUTO_REUSE, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name, reuse=reuse):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def conv2d(input_, output_dim,
           k_h=4, k_w=4, d_h=1, d_w=1, stddev=0.02,
           name="conv2d", reuse=False, padding='same'):
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

        biases = tf.get_variable('biases', [output_dim],
                                 initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])

        return conv

def relu(x):
    return tf.nn.relu(x)

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)



def MaxPooling(x, shape, stride=None, padding='VALID'):
  """
  MaxPooling on images.
  :param input: NHWC tensor.
  :param shape: int or [h, w]
  :param stride: int or [h, w]. default to be shape.
  :param padding: 'valid' or 'same'. default to 'valid'
  :returns: NHWC tensor.
  """
  padding = padding.upper()
  shape = shape4d(shape)
  if stride is None:
    stride = shape
  else:
    stride = shape4d(stride)

  return tf.nn.max_pool(x, ksize=shape, strides=stride, padding=padding)


def shape4d(a):
  # for use with tensorflow
  return [1] + shape2d(a) + [1]


def shape2d(a):
  """
  a: a int or tuple/list of length 2
  """
  if type(a) == int:
      return [a, a]
  if isinstance(a, (list, tuple)):
      assert len(a) == 2
      return list(a)
  raise RuntimeError("Illegal shape: {}".format(a))