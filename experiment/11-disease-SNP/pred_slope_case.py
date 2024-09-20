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

from model.EMO_slope import *
from src.dataGenerator_slope import dataGenerator

def softmax(vec):
    """Compute the softmax in a numerically stable way."""
    vec = vec - np.max(vec)  # softmax(x) = softmax(x+c)
    exp_x = np.exp(vec)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def evaluate(target_model, data):
	_, acc = target_model.evaluate(data)
	print("Restore model, accuracy: {:5.2f}%".format(100*acc))

def predicting(model_size): 

    
    
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

    case_data1 = pd.read_pickle('experiments/11-disease-SNP/data/case/rs1893592_large_atac_t0.dataset')
    testGenerator = dataGenerator(case_data1, batch_size, model_size)

    model.load_weights('model/weights_slope/' + model_size + '/' + model_size + '_trained_weights.tf').expect_partial()
    results = model.predict(testGenerator.generate_validation()[0], batch_size=batch_size)
    print(results)
    
    # save prediction output
    np.save('experiments/11-disease-SNP/data/case/predict.npy', results) 

if __name__=='__main__':

    #predicting('small')
    #predicting('middle')
    predicting('large')
    #predicting('huge') # need large cpu memory