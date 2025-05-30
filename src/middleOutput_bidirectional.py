# generate middle output of EMO layers

import argparse
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

import os
import gc
import copy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from model.EMO import *
from src.dataGenerator import dataGenerator

def generate_middle_output(model_size, layer_name): 

    train_data = pd.read_pickle('datasets/' + model_size + '/train_' + model_size + '_post.pkl')
    
    # check model size
    if(model_size == 'small'):
        batch_size = 16
        model = build_EMO_small()
    elif(model_size == 'middle'):
        batch_size = 16
        model = build_EMO_middle()
    elif(model_size == 'large'):
        batch_size = 16
        model = build_EMO_large()
    elif(model_size == 'huge'):
        batch_size = 16
        model = build_EMO_huge()
    model.summary()

    trainGenerator = dataGenerator(train_data, batch_size, model_size)

    model.load_weights('model/weights/' + model_size + '/' + model_size + '_trained_weights.tf').expect_partial()

    input_features = trainGenerator.generate_validation()[0]
    input_features[4] = np.expand_dims(input_features[4],axis=-1)
    input_features[2] = np.expand_dims(input_features[2],axis=-1)
    npy_input = np.concatenate([input_features[3], input_features[4]], axis=-1)
    npy_input2 = np.concatenate([input_features[0], input_features[1],input_features[2]], axis=-1)

    npy_input = np.sum(npy_input, axis=1)
    npy_input2 = np.sum(npy_input2, axis=1)

    layer_output = tf.keras.models.Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
    npy_out = layer_output.predict(trainGenerator.generate_validation()[0], batch_size=batch_size)  
    npy_out = np.sum(npy_out, axis=1)
    #print(npy_out.shape)

    label = trainGenerator.generate_validation()[1]
    
    # save prediction output
    # npy_input = np.sum(npy_input, axis=1) # for large and huge model
    np.save('model/middle_output_bidirectional/' + model_size + '/' + 'input.npy', npy_input)
    np.save('model/middle_output_bidirectional/' + model_size + '/' + 'input2.npy', npy_input2)
    np.save('model/middle_output_bidirectional/' + model_size + '/' + 'output.npy', npy_out)
    np.save('model/middle_output_bidirectional/' + model_size + '/' + 'label.npy', label)

    # Clear memory and TensorFlow graph
    del model, layer_output, trainGenerator, input_features, label, npy_out
    tf.keras.backend.clear_session()
    gc.collect()


if __name__=='__main__':

    tissue_list = ['Adipose_Subcutaneous','Artery_Tibial','Breast_Mammary_Tissue','Colon_Transverse','Nerve_Tibial','Thyroid']
    
    for tissue in tissue_list:
        generate_middle_output('small', 'bidirectional', tissue)
    for tissue in tissue_list:
        generate_middle_output('middle', 'bidirectional', tissue)
    for tissue in tissue_list:
        generate_middle_output('large', 'bidirectional', tissue)
