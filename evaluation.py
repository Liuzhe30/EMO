# evaluation

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

def softmax(vec):
    """Compute the softmax in a numerically stable way."""
    vec = vec - np.max(vec)  # softmax(x) = softmax(x+c)
    exp_x = np.exp(vec)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def evaluate(target_model, data):
	_, acc = target_model.evaluate(data)
	print("Restore model, accuracy: {:5.2f}%".format(100*acc))

def predicting(model_size, pattern): 
    # pattern: '' or 'rev_'

    test_data = pd.read_pickle('datasets/' + model_size + '/test_' + pattern + model_size + '.pkl')
    
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

    testGenerator = dataGenerator(test_data, batch_size, model_size)

    model.load_weights('model/weights/' + model_size + '/' + model_size + '_trained_weights.tf').expect_partial()
    results = model.predict(testGenerator.generate_validation()[0], batch_size=batch_size)
    label = testGenerator.generate_validation()[1]
    
    # save prediction output
    np.save('model/pred_results/' + model_size + '/' + pattern + 'predict.npy', results)
    np.save('model/pred_results/' + model_size + '/' + pattern + 'label.npy', label)
    

if __name__=='__main__':

    predicting('small','')
    predicting('small','rev_')