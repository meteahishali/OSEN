import os
import numpy as np
import tensorflow as tf
import argparse

from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.ops.array_ops import ones

# Functions and layers.
from transform import affine_grid_generator
from transform import bilinear_sampler
import utils
import networks

# INITIALIZATION
# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('--method', default='OSEN1', help = "OSEN1, NCL_OSEN1, OSEN1_local, CSEN1, OSEN2, NCL_OSEN2, OSEN2_local, CSEN2.")
ap.add_argument('--q', default = 3, help = "Order of the OSEN.")
ap.add_argument('--weights', default = False, help="Evaluate the model.")
ap.add_argument('--mr', default = 0.05, help="Measurement rate.")
ap.add_argument('--epochs', default = 100)
ap.add_argument('--gpu', default = '0', help="GPU id.")
ap.add_argument('--noise', default = 'False', help="True or False.")
ap.add_argument('--batchSize', default = 32)
ap.add_argument('--seed', default = 10, help="Random seed.")
args = vars(ap.parse_args())

param = {}

param['noise'] = args['noise']
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

if param['noise'] == 'True': weightsDir = 'weights_mnist/noisy/'
else: weightsDir = 'weights_mnist/'

if not os.path.exists(weightsDir): os.makedirs(weightsDir)

data = utils.loadData(param) # Modify here. Load the data.

data = utils.takeMeas(data, param) # Take the measurents.

input_shape = data['x_train'][0].shape[1::]

F1score = np.zeros([len(data['x_train']), 1])
F2score = np.zeros([len(data['x_train']), 1])
precision = np.zeros([len(data['x_train']), 1])
specificity = np.zeros([len(data['x_train']), 1])
recall = np.zeros([len(data['x_train']), 1])
CE = np.zeros([len(data['x_train']), 1])
TPR = np.zeros([len(data['x_train']), 1])
FPR = np.zeros([len(data['x_train']), 1])

for i in range(0, len(data['x_train'])):

  print('Split ' + str(i + 1) + ' ... \n\n')

  if 'OSEN' in modelType:
    weightName = weightsDir + modelType + '_q_' + str(q) + '_cr_' + str(param['MeasurementRate']) + '_nuR_' + str(i + 1) + '.h5'
    if 'NCL' in modelType:
      m_meas = data['M_norm'][0].shape[0]
      # If Compressive-Learning is selected, change inputs with the measurements.
      data['x_train'] = data['x_train_meas']
      data['x_val'] = data['x_val_meas']
      data['x_test'] = data['x_test_meas']
      model = eval('networks.' + modelType + '(' + str(m_meas) + ',' + str(input_shape) + ',' + str(q) + ')')

      model.layers[1].weights[0][:, :, 0].assign(data['M_norm'][i])  # Denoiser layer.

      '''
      y = model.predict(data['x_test'][0])
      import matplotlib.pyplot as plt
      plt.imshow(y[0])
      plt.savefig('y1.png')

      plt.imshow(data['y_test'][0][0,:,:,:])
      plt.savefig('x1.png')
      '''

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
    history = model.fit(data['x_train'][i], data['y_train'][i], batch_size = batchSize,
              callbacks=callbacks_model, shuffle=True,
              validation_data=(data['x_val'][i], data['y_val'][i]), epochs = epochs)
    
    
  model.load_weights(weightName)

  precision[i], specificity[i], recall[i], F1score[i], F2score[i], CE[i], TPR[i], FPR[i] = utils.calculatePerformance(
                                                                      param, model, data['x_test'][i], data['y_test'][i],
                                                                      data['x_val'][i], data['y_val'][i])


print('Means:')
print('Obtained Test F1 score: ', np.mean(F1score))
print('Obtained Test F2 score: ', np.mean(F2score))
print('Obtained Test Precision: ', np.mean(precision))
print('Obtained Test Specificity: ', np.mean(specificity))
print('Obtained Test Recall: ', np.mean(recall))
print('Obtained Test CE: ', np.mean(CE))
print('Obtained Test TPR: ', np.mean(TPR))
print('Obtained Test FPR: ', np.mean(FPR))

print('Std:')
print('Obtained Test F1 score: ', np.std(F1score))
print('Obtained Test F2 score: ', np.std(F2score))
print('Obtained Test Precision: ', np.std(precision))
print('Obtained Test Specificity: ', np.std(specificity))
print('Obtained Test Recall: ', np.std(recall))
print('Obtained Test CE: ', np.std(CE))
print('Obtained Test TPR: ', np.std(TPR))
print('Obtained Test FPR: ', np.std(FPR))
