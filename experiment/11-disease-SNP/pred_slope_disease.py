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

    snp_list = ['rs2476601','rs3806624', 'rs7731626', 'rs2234067','rs2233424','rs947474','rs3824660','rs968567','rs3218251']

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
    model.load_weights('model/weights_slope/' + model_size + '/' + model_size + '_trained_weights.tf').expect_partial()

    for snp in snp_list:
        case_data1 = pd.read_pickle('experiments/11-disease-SNP/data/atac_mapping_t0/' + snp + '_' + model_size + '.dataset')
        if(len(case_data1) != 0):
            testGenerator = dataGenerator(case_data1, batch_size, model_size)
            results = model.predict(testGenerator.generate_validation()[0], batch_size=batch_size)
            print(results)
            
            # save prediction output
            np.save('experiments/11-disease-SNP/data/predition_results/' + snp + '_' + model_size + '_t0.npy', results) 

            case_data2 = pd.read_pickle('experiments/11-disease-SNP/data/atac_mapping_t24/' + snp + '_' + model_size + '.dataset')
            testGenerator = dataGenerator(case_data2, batch_size, model_size)
            results = model.predict(testGenerator.generate_validation()[0], batch_size=batch_size)
            print(results)
            
            # save prediction output
            np.save('experiments/11-disease-SNP/data/predition_results/' + snp + '_' + model_size + '_t24.npy', results) 

if __name__=='__main__':

    predicting('small')
    #predicting('middle')
    #predicting('large')
    #predicting('huge') # need large cpu memory