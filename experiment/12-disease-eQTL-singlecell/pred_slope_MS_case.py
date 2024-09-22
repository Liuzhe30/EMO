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

    cell_list = ['B_Naive','CD4_memory','CD4_naive','CD8_memory','Dendritic_Cell','Natural_killer_Cell']
    
    for cell in cell_list:
        print(cell)
        model.load_weights('model/weights_scfinetune_slope/' + model_size + '/' + model_size + '_' + cell + '_trained_weights.tf').expect_partial()
        case_data = pd.read_pickle('experiments/12-disease-eQTL-singlecell/data/' + 'rs1465697_large_' + cell + '.dataset')
        testGenerator = dataGenerator(case_data, batch_size, model_size)
        results = model.predict(testGenerator.generate_validation()[0], batch_size=batch_size)
        print(results)
        # save prediction output
        np.save('experiments/12-disease-eQTL-singlecell/data/rs1465697_large_predict_' + cell + '.npy', results) 

if __name__=='__main__':
    predicting('large') 
