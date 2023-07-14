import tensorflow as tf
import numpy as np

import utils
from layers import SuperOper2DTranspose, SuperOper2D, Oper2D, Oper2DTranspose, GOP
np.random.seed(10)
tf.random.set_seed(10)

### OSEN1
def OSEN1(input_shape, q):
  input = tf.keras.Input(input_shape, name='input')
  x_0 = SuperOper2D(48, 3, activation = 'tanh', q = q)(input)
  x_0 = SuperOper2D(24, 3, activation = 'tanh', q = q)(x_0)
  y = SuperOper2D(2, 3, activation = 'sigmoid', q = q)(x_0)

  model = tf.keras.models.Model(input, y, name='OSEN1')
  model.summary()

  optimizer = tf.keras.optimizers.Adam(lr=0.001)

  model.compile(optimizer = optimizer, loss = utils.loss_func)

  return model

### OSEN2
def OSEN2(input_shape, q):
  
  input = tf.keras.Input(input_shape, name='input')
  x_0 = SuperOper2D(48, 3, activation = 'tanh', q = q)(input)
  x_0 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_0)
  x_0 = SuperOper2DTranspose(24, 3, strides=(2, 2), activation = 'tanh', q = q)(x_0)
  x_0 = SuperOper2D(24, 3, activation = 'tanh', q = q)(x_0)
  y = SuperOper2D(2, 3, activation = 'sigmoid', q = q)(x_0)

  model = tf.keras.models.Model(input, y, name='OSEN2')
  model.summary()

  optimizer = tf.keras.optimizers.Adam(lr=0.001)

  model.compile(optimizer = optimizer, loss = utils.loss_func)

  return model

### OSEN1 with localized kernels.
def OSEN1_local(input_shape, q):
  input = tf.keras.Input(input_shape, name='input')
  x_0 = Oper2D(48, 3, activation = 'tanh', q = q)(input)
  x_0 = Oper2D(24, 3, activation = 'tanh', q = q)(x_0)
  y = Oper2D(2, 3, activation = 'sigmoid', q = q)(x_0)

  model = tf.keras.models.Model(input, y, name='OSEN1_local')
  model.summary()

  optimizer = tf.keras.optimizers.Adam(lr=0.001)

  model.compile(optimizer = optimizer, loss = utils.loss_func)

  return model

### OSEN2 with localized kernels.
def OSEN2_local(input_shape, q):
  input = tf.keras.Input(input_shape, name='input')
  x_0 = Oper2D(48, 3, activation = 'tanh', q = q)(input)
  x_0 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_0)
  x_0 = Oper2DTranspose(24, 3, strides=(2, 2), activation = 'tanh', q = q)(x_0)
  x_0 = Oper2D(24, 3, activation = 'tanh', q = q)(x_0)
  y = Oper2D(2, 3, activation = 'sigmoid', q = q)(x_0)

  model = tf.keras.models.Model(input, y, name='OSEN2_local')
  model.summary()

  optimizer = tf.keras.optimizers.Adam(lr=0.001)

  model.compile(optimizer = optimizer, loss = utils.loss_func)

  return model
  
### CSEN1
def CSEN1(input_shape):
  input = tf.keras.layers.Input(shape = input_shape, name='input')
  x_0 = tf.keras.layers.Conv2D(48, 3, padding = 'same', activation = 'relu')(input)
  x_0 = tf.keras.layers.Conv2D(24, 3, padding = 'same', activation = 'relu')(x_0)
  y = tf.keras.layers.Conv2D(2, 3, padding = 'same', activation = 'relu')(x_0)

  model = tf.keras.models.Model(input, y, name='CSEN1')
  model.summary()

  optimizer = tf.keras.optimizers.Adam(lr=0.001)

  model.compile(optimizer = optimizer, loss = utils.loss_func)

  return model

### CSEN2
def CSEN2(input_shape):
  input = tf.keras.layers.Input(shape = input_shape, name='input')
  x_0 = tf.keras.layers.Conv2D(48, 3, padding = 'same', activation = 'relu')(input)
  x_0 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_0)
  x_0 = tf.keras.layers.Conv2DTranspose(24, 3, strides=(2, 2), padding='same', activation = 'relu')(x_0)
  x_0 = tf.keras.layers.Conv2D(24, 3, padding = 'same', activation = 'relu')(x_0)
  y = tf.keras.layers.Conv2D(2, 3, padding = 'same', activation = 'relu')(x_0)

  model = tf.keras.models.Model(input, y, name='CSEN2')
  model.summary()

  optimizer = tf.keras.optimizers.Adam(lr=0.001)

  model.compile(optimizer = optimizer, loss = utils.loss_func)

  return model