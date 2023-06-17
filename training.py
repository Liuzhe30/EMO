# training

import argparse
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

import os
import copy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from model.EMO import *
from src.dataGenerator import dataGenerator

# for gpu training
# os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_size', default='small')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.005, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.05, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs") 
    parser.add_argument('--save_dir', default='model/weights/')
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    args = parser.parse_args()
    model_size = args.model_size

    # load datasets
    train_pkl = pd.read_pickle('datasets/' + model_size + '/train_' + model_size + '.pkl')
    test_pkl = pd.read_pickle('datasets/' + model_size + '/test_' + model_size + '.pkl')

    # split train/validation
    train_data = train_pkl[0:int(train_pkl.shape[0]*float(8/9))].reset_index(drop=True)
    valid_data = train_pkl[int(train_pkl.shape[0]*float(8/9)):].reset_index(drop=True)
    test_data = test_pkl

    # check model size
    if(model_size == 'small'):
        batch_size = 32
        model = build_EMO_small()
    elif(model_size == 'middle'):
        batch_size = 32
        model = build_EMO_middle()
    elif(model_size == 'large'):
        batch_size = 16
        model = build_EMO_large()
    elif(model_size == 'huge'):
        batch_size = 4
        model = build_EMO_huge()

    model.summary()
    
    # callbacks
    log = tf.keras.callbacks.CSVLogger(args.save_dir + model_size + '/log.csv')
    #tensorboard = tf.keras.callbacks.TensorBoard(log_dir=args.save_dir + model_size + '/tensorboard-logs', histogram_freq=int(args.debug))
    #EarlyStopping = callbacks.EarlyStopping(monitor='val_cc2', min_delta=0.01, patience=5, verbose=0, mode='max', baseline=None, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(args.save_dir + args.model_size + '/' + args.model_size + '_trained_weights.tf', monitor='val_acc', mode='max', #val_categorical_accuracy val_acc
                                       save_best_only=True, save_weights_only=True, verbose=1)        
    #lr_decay = tf.keras.callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # Train the model and save it
    model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['mae', 'acc'])
    
    trainGenerator = dataGenerator(train_data, batch_size, model_size)
    validGenerator = dataGenerator(valid_data, batch_size, model_size)
    
    history = model.fit(trainGenerator.generate_batch(), # Tf2 new feature
          steps_per_epoch = len(train_data)/batch_size,
          epochs = args.epochs, verbose=1,
          validation_data = validGenerator.generate_batch(),
          validation_steps = len(valid_data)/batch_size,
          #callbacks = [log, tensorboard, checkpoint, lr_decay],
          callbacks = [log, checkpoint],
          shuffle = True,
          #class_weight = class_weights,
          #batch_size=args.batch_size,
          workers = 1).history

    print('Trained model saved to \'%s/trained_model.tf\'' % args.save_dir)
    
    
    #dataGenerator = dataGenerator(train_data, batch_size, model_size)