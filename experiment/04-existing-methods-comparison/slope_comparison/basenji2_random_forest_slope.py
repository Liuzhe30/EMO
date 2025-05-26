# random forest for training basenji2-regression model

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

compare_tissue_list = ['Adipose_Subcutaneous', 'Artery_Tibial', 'Breast_Mammary_Tissue',
                       'Colon_Transverse', 'Nerve_Tibial', 'Thyroid']
model_size_list = ['small', 'middle']

final_path = 'basenji2_results_slope/'
output_path = 'basenji2_prediction_results_slope/'

for tissue in compare_tissue_list:
    for model_size in model_size_list:
        # Load training data
        train_table = pd.read_pickle(final_path + 'train_' + model_size + '_' + tissue + '_slope.pkl')
        print(train_table.head())

        # Prepare features and labels
        feature_list = []
        labels = np.array(train_table['slope'].astype("float"))  # Regression uses float labels
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

        # Train regression model
        reg = RandomForestRegressor()
        reg.fit(X_train, Y_train)

        # Load test data
        test_table = pd.read_pickle(final_path + 'test_' + model_size + '_' + tissue + '_slope.pkl')
        feature_list = []
        labels = np.array(test_table['slope'].astype("float"))
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

        # Predict
        y_pred = reg.predict(X_test)

        # Save results
        np.save(output_path + model_size + '_' + tissue + '_label.npy', Y_test)
        np.save(output_path + model_size + '_' + tissue + '_prediction.npy', y_pred)
