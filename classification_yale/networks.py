import tensorflow as tf
import numpy as np

from layers import SuperOper2D, Oper2D, GOP
import tensorflow_addons as tfa
np.random.seed(10)
tf.random.set_seed(10)

### OSEN Classification
def OSEN_C(input_shape, q):

  norm = tfa.layers.InstanceNormalization

  input = tf.keras.Input(input_shape, name='input')
  x_0 = SuperOper2D(48, 3, activation = 'tanh', q = q)(input)
  x_0 = norm()(x_0)
  x_0 = SuperOper2D(24, 3, activation = 'tanh', q = q)(x_0)
  x_0 = norm()(x_0)
  s_0 = SuperOper2D(1, 3, activation = 'sigmoid', q = q, name='support')(x_0)
  x_0 = tf.keras.layers.AveragePooling2D(pool_size=(8, 4))(s_0) # Sparse code shapes.
  x_0 = norm()(x_0)
  y = tf.keras.layers.Flatten()(x_0)
  y = tf.keras.layers.Softmax(name='classification')(y)

  model = tf.keras.models.Model(input, [y, s_0],name='OSEN1_C')
  model.summary()

  optimizer = tf.keras.optimizers.Adam(lr=0.001)

  model.compile(optimizer=optimizer,
                loss = {'classification': tf.keras.losses.CategoricalCrossentropy(),
                        'support': tf.keras.losses.MeanAbsoluteError()},
                metrics = {'classification': 'accuracy', 'support': 'mae'},
                loss_weights = {"classification": 1.0, "support": 1.0})

  return model

### NCL_OSEN Classification
def NCL_OSEN_C(m_meas, input_shape, q):

  M = input_shape[0]
  N = input_shape[1]

  norm = tfa.layers.InstanceNormalization

  input = tf.keras.Input(m_meas, name='input')
  x_0 = GOP(M * N, activation = 'tanh', q = q)(input)
  x_0 = tf.keras.layers.Reshape(input_shape)(x_0) # Size of reshaped proxy from CRC estimation.

  x_0 = SuperOper2D(48, 3, activation = 'tanh', q = q)(x_0)
  x_0 = norm()(x_0)
  x_0 = SuperOper2D(24, 3, activation = 'tanh', q = q)(x_0)
  x_0 = norm()(x_0)
  s_0 = SuperOper2D(1, 3, activation = 'sigmoid', q = q, name='support')(x_0)
  x_0 = tf.keras.layers.AveragePooling2D(pool_size=(8, 4))(s_0) # Sparse code shapes.
  x_0 = norm()(x_0)
  y = tf.keras.layers.Flatten()(x_0)
  y = tf.keras.layers.Softmax(name='classification')(y)

  model = tf.keras.models.Model(input, [y, s_0],name='NCL_OSEN1_C')
  model.summary()

  optimizer = tf.keras.optimizers.Adam(lr=0.001)

  model.compile(optimizer=optimizer,
                loss = {'classification': tf.keras.losses.CategoricalCrossentropy(),
                        'support': tf.keras.losses.MeanAbsoluteError()},
                metrics = {'classification': 'accuracy', 'support': 'mae'},
                loss_weights = {"classification": 1.0, "support": 1.0})

  return model

# OSEN Classification with localized kernels.
def OSEN_C_local(input_shape, q):

  norm = tfa.layers.InstanceNormalization

  input = tf.keras.Input(input_shape, name='input')
  x_0 = Oper2D(48, 3, activation = 'tanh', q = q)(input)
  x_0 = norm()(x_0)
  x_0 = Oper2D(24, 3, activation = 'tanh', q = q)(x_0)
  x_0 = norm()(x_0)
  s_0 = Oper2D(1, 3, activation = 'sigmoid', q = q, name='support')(x_0)
  x_0 = tf.keras.layers.AveragePooling2D(pool_size=(8, 4))(s_0) # Sparse code shapes.
  x_0 = norm()(x_0)
  y = tf.keras.layers.Flatten()(x_0)
  y = tf.keras.layers.Softmax(name='classification')(y)

  model = tf.keras.models.Model(input, [y, s_0],name='OSEN1_C_local')
  model.summary()

  optimizer = tf.keras.optimizers.Adam(lr=0.001)

  model.compile(optimizer=optimizer,
                loss = {'classification': tf.keras.losses.CategoricalCrossentropy(),
                        'support': tf.keras.losses.MeanAbsoluteError()},
                metrics = {'classification': 'accuracy', 'support': 'mae'},
                loss_weights = {"classification": 1.0, "support": 1.0})

  return model

# CSEN Classification
def CSEN_C(input_shape):
  input = tf.keras.Input(input_shape, name='input')
  x_0 = tf.keras.layers.Conv2D(48, 3, padding = 'same', activation='relu')(input)
  x_0 = tf.keras.layers.Conv2D(24, 3, padding = 'same', activation='relu')(x_0)
  x_0 = tf.keras.layers.Conv2D(1, 3, padding = 'same', activation='relu')(x_0)
  x_0 = tf.keras.layers.AveragePooling2D(pool_size=(8, 4))(x_0) # Sparse code shapes.
  y = tf.keras.layers.Flatten()(x_0)
  y = tf.keras.layers.Softmax()(y)

  model = tf.keras.models.Model(input, y, name='CSEN1_C')
  model.summary()

  optimizer = tf.keras.optimizers.Adam(lr=0.001)
  model.compile(optimizer=optimizer,
                loss = 'categorical_crossentropy',
                metrics=['accuracy'])

  return model