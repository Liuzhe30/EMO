# training

import argparse
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from model.EMO import *
from src.dataGenerator import dataGenerator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_size', default='small')
    parser.add_argument('-a', '--augmentation', default='False')
    args = parser.parse_args()
    model_size = args.model_size
    augmentation = args.augmentation

    # load datasets
    if(augmentation == 'False'):
        train_pkl = pd.read_pickle('datasets/' + model_size + '/train_' + model_size + '.pkl')
    elif(augmentation == 'True'):
        train_pkl = pd.read_pickle('datasets/' + model_size + '/train_aug_' + model_size + '.pkl')
    test_pkl = pd.read_pickle('datasets/' + model_size + '/test_' + model_size + '.pkl')

    # split train/validation
    train_data = train_pkl[0:int(train_pkl.shape[0]*float(8/9))].reset_index(drop=True)
    valid_data = train_pkl[int(train_pkl.shape[0]*float(8/9)):].reset_index(drop=True)
    test_data = test_pkl

    # check model size
    if(model_size == 'small'):
        batch_size = 128
        model = build_EMO_small()
        model.summary()
        #dataGenerator = dataGenerator(train_data, batch_size, model_size)