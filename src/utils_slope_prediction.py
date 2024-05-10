import sys 
sys.path.append("..") 
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from model.EMO_slope import *
from src.dataGenerator_slope import dataGenerator

def get_slope_prediction_result(input_variant, TSS_distance, atac_variant, atac_between, genome_path, weights_path):
    
    # 1 prepare dataset
    chr_str = input_variant.split('_')[0]
    position = int(input_variant.split('_')[1])
    before_mutation = input_variant.split('_')[2]
    after_mutation = input_variant.split('_')[3]
    tss_position = position - TSS_distance

    with open(genome_path + chr_str + '.fasta') as fa:
        line = fa.readline()
        variant_51_seq = line[position - 26:position + 25]
        if(TSS_distance > 0):
            seq_between_variant_tss = line[tss_position - 1:position]
        else:
            seq_between_variant_tss = line[position -1: tss_position]
        if(line[position - 1] >= 'a' and line[position - 1] <= 'z'):
            variant_51_seq_after_mutation = line[position - 26: position - 1] + after_mutation.lower() + line[position: position + 25]
        else:
            variant_51_seq_after_mutation = line[position - 26: position - 1] + after_mutation + line[position: position + 25]
    
    test_data = pd.DataFrame(columns=['variant_51_seq', 'variant_51_seq_after_mutation', 'atac_variant_51', 'seq_between_variant_tss', 'atac_between', 'label'])
    test_data = test_data._append([{'variant_51_seq':variant_51_seq, 'variant_51_seq_after_mutation':variant_51_seq_after_mutation, 
                                    'atac_variant_51':atac_variant, 'seq_between_variant_tss':seq_between_variant_tss,
                                    'atac_between':atac_between,'slope':0}], ignore_index=True)

    # 2 prediction
    model_size = get_model_size(TSS_distance)
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
    #model.summary()
    testGenerator = dataGenerator(test_data, batch_size, model_size)
    model.load_weights(weights_path + model_size + '_slope_weights.tf').expect_partial()
    results = model.predict(testGenerator.generate_validation()[0], batch_size=batch_size)
    return np.abs(results[0][0])

def get_model_size(TSS_distance):
    tss_distance = np.abs(TSS_distance)
    if(tss_distance < 1000):
        model_size = 'small'
    elif(tss_distance >= 1000 and tss_distance < 10000):
        model_size = 'middle'
    elif(tss_distance >= 10000 and tss_distance < 100000):
        model_size = 'large'
    elif(tss_distance >= 100000):
        model_size = 'huge'
    return model_size

