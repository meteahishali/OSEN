import os
import numpy as np
import tensorflow as tf
import argparse

# Functions and layers.
import utils
import networks

# INITIALIZATION
# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('--method', default='OSEN_C', help = "OSEN_C, NCL_OSEN_C, OSEN_C_local, CSEN_C.")
ap.add_argument('--q', default = 1, help = "Order of the OSEN.")
ap.add_argument('--weights', default = False, help="Evaluate the model.")
ap.add_argument('--mr', default = 0.01, help="Measurement rate: 0.01, 0.05, 0.25.")
ap.add_argument('--epochs', default = 30)
ap.add_argument('--gpu', default = '0', help="GPU id.")
ap.add_argument('--batchSize', default = 5)
ap.add_argument('--seed', default = 10, help="Random seed.")
args = vars(ap.parse_args())

param = {}

modelType = args['method'] # CSEN and OSEN.
weights = args['weights'] # True or False.
q = args['q'] # The order of the OSEN.
param['MeasurementRate'] = float(args['mr'])
epochs = int(args['epochs'])
batchSize = int(args['batchSize'])
seed = int(args['seed'])
os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']

np.random.seed(seed)
tf.random.set_seed(seed)

weightsDir = 'weights_yale_b/'

if not os.path.exists(weightsDir): os.makedirs(weightsDir)

data = utils.loadData(param) # Modify here. Load the data.

input_shape = data['x_train'][0].shape[1::]

accuracy = np.zeros([len(data['x_train']), 1])

for i in range(0, len(data['x_train'])):

  print('Split ' + str(i + 1) + ' ... \n\n')

  if 'OSEN' in modelType:
    weightName = weightsDir + modelType + '_q_' + str(q) + '_cr_' + str(param['MeasurementRate']) + '_nuR_' + str(i + 1) + '.h5'
    if 'CL' in modelType:
      m_meas = data['M_norm'][0].shape[0]
      # If Compressive-Learning is selected, change inputs with the measurements.
      data['x_train'] = data['x_train_meas']
      data['x_val'] = data['x_val_meas']
      data['x_test'] = data['x_test_meas']
      model = eval('networks.' + modelType + '(' + str(m_meas) + ',' + str(input_shape) + ',' + str(q) + ')')

      model.layers[1].weights[0][:, :, 0].assign(data['M_norm'][i])  # Denoiser layer.

    else:
      model = eval('networks.' + modelType + '(' + str(input_shape) + ',' + str(q) + ')')
  else:
    weightName = weightsDir + modelType + '_cr_' + str(param['MeasurementRate']) + '_nuR_' + str(i + 1) + '.h5'
    model = eval('networks.' + modelType + '(' + str(input_shape) + ')')

  checkpoint_model = tf.keras.callbacks.ModelCheckpoint(
      weightName, monitor='val_loss', verbose=1,
      save_best_only=True, mode='min', save_weights_only=True)

  callbacks_model = [checkpoint_model]

  if weights is False:
    len(model.output) # Applies both support estimation and classification.
    history = model.fit(data['x_train'][i],
              {'classification': data['y_train'][i], 'support': data['m_train'][i]},
              batch_size = batchSize, callbacks=callbacks_model, shuffle=True, epochs = epochs, 
              validation_data=(data['x_val'][i], {'classification': data['y_val'][i], 'support': data['m_val'][i]}))
    
    
  model.load_weights(weightName)

  accuracy[i] = utils.calculatePerformance(param, model, data['x_test'][i], data['y_test'][i], data['x_val'][i], data['y_val'][i])

for i in range(0, len(accuracy)):
  print('Accuracy for set ' + str(i + 1) + ': ' + str(accuracy[i]))
print('Mean accuracy: ', np.mean(accuracy))
print('Std accuracy: ', np.std(accuracy))