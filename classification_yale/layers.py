import tensorflow as tf

import transform as ts


####################################################
# Operational Layers with Super (Generative) Neurons
class SuperOper2D(tf.keras.Model):
  def __init__(self, filters, kernel_size, activation = None, q = 1, name = '', shift_init = 'zeros'):
    super(SuperOper2D, self).__init__(name=name)

    self.activation = activation
    self.q = q
    self.all_layers = []

    if shift_init == 'uniform':
      txy = tf.random.uniform(shape = [filters, 2, 1], minval=-0.2, maxval=0.2)
    else:
      txy = tf.zeros([filters, 2, 1], dtype=tf.float32)
      
    self.txy = tf.Variable(txy, trainable = True)

    for i in range(0, q):  # q convolutional layers.
      self.all_layers.append(tf.keras.layers.Conv2D(filters,
                                                    (kernel_size,
                                                      kernel_size),
                                                    padding='same', activation=None))

  @tf.function
  def call(self, input_tensor, training=False):
    
    def shift(feat_maps):
      def func(elems):
        tensorr, txy = elems
      
        A = tf.concat([tf.eye(2), txy], axis=-1)
        A = tf.expand_dims(A, axis=0)

        grids = ts.affine_grid_generator(M, N, A)

        x_s = grids[0, :, :]
        y_s = grids[1, :, :]

        shifted_tensor = ts.bilinear_sampler(tensorr, x_s, y_s)

        return shifted_tensor, txy

      shifted_tensors, _ = tf.vectorized_map(fn=func, elems = (feat_maps, self.txy))
      return shifted_tensors
      #return tf.map_fn(lambda input: func(input[0], input[1]), (feat_maps, self.txy))

    x = self.all_layers[0](input_tensor)  # First convolutional layer.

    if self.q > 1:
      for i in range(1, self.q):
        x += self.all_layers[i](tf.math.pow(input_tensor, i + 1))

    M = x.shape[1]
    N = x.shape[2]

    x = tf.keras.layers.Permute((3,1,2))(x)
    x = tf.vectorized_map(fn=shift, elems = x)
    x = tf.keras.layers.Permute((2,3,1))(x)
    
    if self.activation is not None:
      return eval('tf.nn.' + self.activation + '(x)')
    else:
      return x

####################################################
# Operational Layers with localized kernels.
class Oper2D(tf.keras.Model):
  def __init__(self, filters, kernel_size, activation = None, q = 1, name = ''):
    super(Oper2D, self).__init__(name=name)

    self.activation = activation
    self.q = q
    self.all_layers = []

    for i in range(0, q):  # q convolutional layers.
      self.all_layers.append(tf.keras.layers.Conv2D(filters,
                                                    (kernel_size,
                                                      kernel_size),
                                                    padding='same', activation=None))

  @tf.function
  def call(self, input_tensor, training=False):
    
    x = self.all_layers[0](input_tensor)  # First convolutional layer.

    if self.q > 1:
      for i in range(1, self.q):
        x += self.all_layers[i](tf.math.pow(input_tensor, i + 1))
    
    if self.activation is not None:
      return eval('tf.nn.' + self.activation + '(x)')
    else:
      return x

####################################################
# Transposed Layers.
####################################################

####################################################
# Transposed Operational Layers for upsampling with Super (Generative) Neurons
class SuperOper2DTranspose(tf.keras.Model):
  def __init__(self, filters, kernel_size, strides = (1, 1), activation = None, q = 1, name = '', shift_init = 'zeros'):
    super(SuperOper2DTranspose, self).__init__(name=name)

    self.activation = activation
    self.q = q
    self.all_layers = []

    if shift_init == 'uniform':
      txy = tf.random.uniform(shape = [filters, 2, 1], minval=-0.2, maxval=0.2)
    else:
      txy = tf.zeros([filters, 2, 1], dtype=tf.float32)
      
    self.txy = tf.Variable(txy, trainable = True)

    for i in range(0, q):  # q convolutional layers.
      self.all_layers.append(tf.keras.layers.Conv2DTranspose(filters,
                                                    kernel_size,
                                                    strides = strides,
                                                    padding='same', activation=None))

  @tf.function
  def call(self, input_tensor, training=False):
    
    def shift(feat_maps):
      def func(elems):
        tensorr, txy = elems
      
        A = tf.concat([tf.eye(2), txy], axis=-1)
        A = tf.expand_dims(A, axis=0)

        grids = ts.affine_grid_generator(M, N, A)

        x_s = grids[0, :, :]
        y_s = grids[1, :, :]

        shifted_tensor = ts.bilinear_sampler(tensorr, x_s, y_s)

        return shifted_tensor, txy

      shifted_tensors, _ = tf.vectorized_map(fn=func, elems = (feat_maps, self.txy))
      return shifted_tensors
      #return tf.map_fn(lambda input: func(input[0], input[1]), (feat_maps, self.txy))
    
    x = self.all_layers[0](input_tensor)  # First convolutional layer.

    if self.q > 1:
      for i in range(1, self.q):
        x += self.all_layers[i](tf.math.pow(input_tensor, i + 1))

    M = x.shape[1]
    N = x.shape[2]

    x = tf.keras.layers.Permute((3,1,2))(x)
    x = tf.vectorized_map(fn=shift, elems = x)
    x = tf.keras.layers.Permute((2,3,1))(x)
    
    if self.activation is not None:
      return eval('tf.nn.' + self.activation + '(x)')
    else:
      return x

####################################################
# Transposed Operational Layers using localized kernels.
class Oper2DTranspose(tf.keras.Model):
  def __init__(self, filters, kernel_size, strides = (1, 1), activation = None, q = 1, name = ''):
    super(Oper2DTranspose, self).__init__(name=name)

    self.activation = activation
    self.q = q
    self.all_layers = []

    for i in range(0, q):  # q convolutional layers.
      self.all_layers.append(tf.keras.layers.Conv2DTranspose(filters,
                                                    kernel_size,
                                                    strides = strides,
                                                    padding='same', activation=None))

  @tf.function
  def call(self, input_tensor, training=False):

    x = self.all_layers[0](input_tensor)  # First convolutional layer.

    if self.q > 1:
      for i in range(1, self.q):
        x += self.all_layers[i](tf.math.pow(input_tensor, i + 1))

    if self.activation is not None:
      return eval('tf.nn.' + self.activation + '(x)')
    else:
      return x

####################################################
# Self-organized Operational Perceptrons (Self-OPS).
class GOP(tf.keras.layers.Layer):
  def __init__(self, num_outputs, activation = None, q = 1, name = ''):
    super(GOP, self).__init__(name=name)
    self.num_outputs = num_outputs
    self.q = q
    self.activation = activation

  def build(self, input_shape):
    self.kernel = self.add_weight("kernel",
                                  shape=[int(input_shape[-1]),
                                          self.num_outputs,
                                          self.q])
    self.bias = self.add_weight("bias", shape=[self.num_outputs, self.q])

  @tf.function
  def call(self, inputs):
    x = tf.matmul(inputs, self.kernel[:, :, 0]) + self.bias[:, 0]
    if self.q > 1:
      for i in range(1, self.q):
        x += tf.matmul(tf.math.pow(inputs, i + 1), self.kernel[:, :, i]) + self.bias[:, i]

    if self.activation is not None:
      return eval('tf.nn.' + self.activation + '(x)')
    else:
      return x