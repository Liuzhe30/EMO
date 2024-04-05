# random forest for training enformer-classification model

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

compare_tissue_list = ['Adipose_Subcutaneous','Artery_Tibial','Breast_Mammary_Tissue','Colon_Transverse','Nerve_Tibial','Testis','Thyroid']
model_size_list = ['small','middle','large']

compare_tissue_list = ['Adipose_Subcutaneous']
model_size_list = ['small']

final_path = '/data/eqtl/datasets/tissue_specific/enformer_final/'
output_path = '../prediction_results/'

for tissue in compare_tissue_list:
    for model_size in model_size_list:
        # training model
        train_table = pd.read_pickle(final_path + 'train_' + model_size + '_' + tissue + '.pkl')
        print(train_table.head())

        feature_list = []
        labels = np.array(train_table['label'].astype("int"))
        for i in range(train_table.shape[0]):
            sample_feature = []
            sample_feature += train_table['result_before'][i].flatten().tolist()
            sample_feature += train_table['result_after'][i].flatten().tolist()
            feature_list.append(sample_feature)
        features = np.array(feature_list)

        X_train = features
        Y_train = labels
        print(X_train.shape)
        print(Y_train.shape)

        clf = RandomForestClassifier()
        clf.fit(X_train,Y_train)

        # prediction output
        test_table = pd.read_pickle(final_path + 'test_' + model_size + '_' + tissue + '.pkl')
        feature_list = []
        labels = np.array(test_table['label'].astype("int"))
        for i in range(test_table.shape[0]):
            sample_feature = []
            sample_feature += test_table['result_before'][i].flatten().tolist()
            sample_feature += test_table['result_after'][i].flatten().tolist()
            feature_list.append(sample_feature)
        features = np.array(feature_list)

        X_test = features
        Y_test = labels
        print(X_test.shape)
        print(Y_test.shape)

        y_score = clf.predict(X_test)                                   
        y_score_pro = clf.predict_proba(X_test) # (.., 2)

        # save results
        np.save(output_path + model_size + '_' + tissue + '_label.npy',Y_test)
        np.save(output_path + model_size + '_' + tissue + '_score.npy',y_score)
        np.save(output_path + model_size + '_' + tissue + '_score_pro.npy',y_score_pro)