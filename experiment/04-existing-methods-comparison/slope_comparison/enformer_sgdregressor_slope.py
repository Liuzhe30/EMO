# SGDRegressor for training enformer-regression model (fast version with L2 regularization)

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
import os

compare_tissue_list = ['Adipose_Subcutaneous', 'Artery_Tibial', 'Breast_Mammary_Tissue',
                       'Colon_Transverse', 'Nerve_Tibial', 'Testis', 'Thyroid']
model_size_list = ['small', 'middle', 'large']

final_path = 'enformer_results_slope/'
output_path = 'enformer_prediction_results_slope/'

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

for tissue in compare_tissue_list:
    for model_size in model_size_list:
        # Load training data
        train_path = final_path + 'train_' + model_size + '_' + tissue + '_slope.pkl'
        train_table = pd.read_pickle(train_path)
        print(train_table.head())

        # Prepare training features and labels
        feature_list = []
        labels = np.array(train_table['slope'].astype("float"))
        for i in range(train_table.shape[0]):
            sample_feature = []
            sample_feature += train_table['result_before'][i].flatten().tolist()
            sample_feature += train_table['result_after'][i].flatten().tolist()
            feature_list.append(sample_feature)
        features = np.array(feature_list)

        X_train = features
        Y_train = labels
        print("Train:", X_train.shape, Y_train.shape)

        # Train model using SGD with L2 regularization
        reg = SGDRegressor(loss='squared_error', penalty='l2', alpha=0.0001,
                           max_iter=1000, tol=1e-3, random_state=42)
        reg.fit(X_train, Y_train)

        # Load test data
        test_path = final_path + 'test_' + model_size + '_' + tissue + '_slope.pkl'
        test_table = pd.read_pickle(test_path)
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
        print("Test:", X_test.shape, Y_test.shape)

        # Predict
        y_pred = reg.predict(X_test)

        # Save results
        np.save(output_path + model_size + '_' + tissue + '_label.npy', Y_test)
        np.save(output_path + model_size + '_' + tissue + '_prediction.npy', y_pred)
        print(f"[✓] Saved: {model_size}_{tissue}")
