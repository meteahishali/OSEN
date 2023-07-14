import os
import numpy as np
import tensorflow as tf
import argparse

# Functions.
import utils
import networks

# INITIALIZATION
# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('--method', default='OSEN1', help = "OSEN1, OSEN2, OSEN1_local, OSEN2_local, CSEN1, CSEN2.")
ap.add_argument('--q', default = 3, help = "Order of the OSEN.")
ap.add_argument('--weights', default = True, help="Evaluate the model.")
ap.add_argument('--mr', default = 0.05, help="Measurement rate.")
ap.add_argument('--epochs', default = 100, help="Number of epochs.")
ap.add_argument('--gpu', default = '0', help="GPU id.")
ap.add_argument('--batchSize', default = 32, help="Batch size.")
ap.add_argument('--seed', default = 10, help="Random seed.")
ap.add_argument('--nu', default = 1, help = "Which run?")
args = vars(ap.parse_args())

param = {}

modelType = args['method'] # CSEN and OSEN.
weights = args['weights'] # True or False.
q = args['q'] # The order of the OSEN.
param['MeasurementRate'] = float(args['mr'])
epochs = int(args['epochs'])
batchSize = int(args['batchSize'])
seed = int(args['seed'])
param['nu'] = int(args['nu'])
os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']

np.random.seed(int(param['nu']) * 10)
tf.random.set_seed(int(param['nu']) * 10)


weightsDir = 'weights_mri/'

if not os.path.exists(weightsDir): os.makedirs(weightsDir)
if not os.path.exists('output/'): os.makedirs('output/')

data = utils.loadData(param) # Modify here. Load the data.

data = utils.takeMeas(data, param) # Take the measurents.

np.save('masks_' + str(param['MeasurementRate']) + '_' + str(param['nu']) + '.npy', data['M_masks']) # Save the measurement masks.

input_shape = data['x_train_d'].shape[1::]

print('Run ' + str(param['nu']) + ' ... \n\n')

if 'OSEN' in modelType:
  weightName = weightsDir + modelType + '_q_' + str(q) + '_cr_' + str(param['MeasurementRate']) + '_nuR_' + str(param['nu']) + '.h5'
  model = eval('networks.' + modelType + '(' + str(input_shape) + ',' + str(q) + ')')
  
else: # CSEN
  weightName = weightsDir + modelType + '_cr_' + str(param['MeasurementRate']) + '_nuR_' + str(param['nu']) + '.h5'
  model = eval('networks.' + modelType + '(' + str(input_shape) + ')')

checkpoint_model = tf.keras.callbacks.ModelCheckpoint(
    weightName, monitor='val_loss', verbose=1,
    save_best_only=True, mode='min', save_weights_only=True)

callbacks_model = [checkpoint_model]

if weights is False:
  history = model.fit(data['x_train_d'], data['y_train_d'], batch_size = batchSize,
        callbacks=callbacks_model, shuffle=True,
        validation_data=(data['x_val_d'], data['y_val_d']), epochs = epochs)
  
  
model.load_weights(weightName)

y_pred_test = model.predict(data['x_test_d'], batch_size=32)

if 'OSEN' in modelType:
  np.save('output/predictions_' + modelType
          + '_q_' + str(q)
          + '_cr_' + str(param['MeasurementRate'])
          + '_nuR_' + str(param['nu']) + '.npy', y_pred_test)
else:
  np.save('output/predictions_' + modelType
          + '_cr_' + str(param['MeasurementRate'])
          + '_nuR_' + str(param['nu']) + '.npy', y_pred_test)

print('Run ' + str(param['nu']) + ' ... \n\n')
precision, specificity, recall, F1score, F2score, CE, TPR, FPR = utils.calculatePerformance(
                                                                    model, data['x_test_d'], data['y_test_d'],
                                                                    data['x_val_d'], data['y_val_d'])