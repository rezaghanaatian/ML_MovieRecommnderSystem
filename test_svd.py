# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 16:48:24 2018

@author: Reza Ghanaatian
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
from surprise.prediction_algorithms import SVD
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split


from helpers import *
#from helpers_v2 import cross_validator

print("reading data...")
path_dataset = "data/data_train.csv"
path_test_dataset = "data/sample_submission.csv"

train_df = load_dataset(path_dataset)
test_df = load_dataset(path_test_dataset)
print("reading data done")
train_df.head()

print("dimention of train is {} by {}".format(train_df.shape[0], train_df.shape[1]))



#%%
def matrix_factorization_svd(train, test, **kwargs):
    """
    matrix factorization using SVD from surprise library
    
    input
        train:      pandas dataframe
        test:       pandas dataframe
        **kwargs:   arbitrary keyword argument
        
    output
        pandas dataframe: prediction
    """
    
    # hyper parameters
    n_epochs = 30#kwargs['n_epochs']
    n_factors = 10#kwargs['n_factors']
    lr_all = 0.001#kwargs['lr_all']
    reg_all = 0.01#kwargs['reg_all']
    
    # set the parameters for SVD
    algo = SVD(n_factors=n_factors,n_epochs=n_epochs,lr_all=lr_all,reg_all=reg_all)
    
    #chnage the column order based on the reader requirment
    train = train[['User','Movie','Prediction']]
    #test = test[['User','Movie','Prediction']]

    #create test set method 2
    reader = Reader(line_format='user item rating', sep=',')
    data = Dataset.load_from_df(train,reader)
    trainset = data.build_full_trainset()
    
    # training
    algo.fit(trainset)
    
    # prediction
    test_sp = df_to_sp(test)
    nz_movie, nz_user = test_sp.nonzero()
    nz_test = list(zip(nz_movie, nz_user))
    for movie, user in nz_test:
        val = algo.predict(user+1,movie+1,verbose=False)
        if val.est > 5:
            pred= 5
        elif val.est < 1:
            pred = 1
        else:
            pred = np.round(val.est)
                
        test_sp[movie, user] = pred
    
    test_pred = sp_to_df(test_sp)
    
    err_ts = compute_error2(test, test_pred)
    print("RMSE on test set: {}.".format(err_ts))
    

    # create testset the other methods 
#    # dump the pandas DF into files to be used in SVD
#    train_file = 'tmp_train.csv'
#    #test_file = 'tmp_test.csv'
#    train.to_csv(train_file, index=False, header=False)
#    #test.to_csv(test_file, index=False, header=False)
#    # create reader object
#    reader = Reader(line_format='user item rating', sep=',')
#    #load the data
#    data = Dataset.load_from_file(train_file, reader)
    
    #trainset, testset = train_test_split(data, test_size=0.001)
    
#    fold = [(train_file, test_file)]
#    data = Dataset.load_from_folds(fold, reader=reader)
#    for trainset, testset in data.folds():
#        # Train
#        algo.train(trainset)
#
#        # Predict
#        predictions = algo.test(testset)
    
    return test_pred

    
#%%
def matrix_factorization_svd_normalized(train, test, **kwargs):
    """
    matrix factorization using SVD from surprise library with normalization
    
    input
        train:      pandas dataframe
        test:       pandas dataframe
        **kwargs:   arbitrary keyword argument
        
    output
        pandas dataframe: prediction
    """
    
     # Normalize the train data
    def dict_mean_user(train):
        train=train.astype(int)
        global_mean = train.groupby('User').mean().Prediction.mean()
        return dict(train.groupby('User').mean().Prediction - global_mean)
    
    def normalize_deviation(df):
        """ Set all the mean to overall mean (BEST METHOD)
        """
        norm_df = pd.DataFrame.copy(df)
        norm_df['Prediction'] = df.apply(
            lambda x: x['Prediction'] - dict_mean_user(df)[x['User']],
            axis=1)

        return norm_df
    
    def recover_deviation(df):
        """ Recover from 'normalized_deviation' table
        """
        recovered_df = pd.DataFrame.copy(df)
        recovered_df['Prediction'] = df.apply(
            lambda x: x['Prediction'] + dict_mean_user(df)[x['User']],
            axis=1)

        return recovered_df
    
    train_normalized = normalize_deviation(train)
    print ("******************")
    print(train_normalized[:10])

    # Predict using the normalized trained data
    test_pred_normalized = matrix_factorization_svd(train_normalized, test)
    
    #rescale the prediction to recover the actual mean
    test_pred = recover_deviation(test_pred_normalized)
       
    err_ts = compute_error2(test, test_pred)
    print("RMSE on test set after rescaling: {}.".format(err_ts))
    
    return test_pred

#%% testing the algorithm    
#%%
train_df2 = train_df.head(1000)
train_df3= train_df.head(100000);

#%%
output = matrix_factorization_svd(train_df2, train_df2)

#%%
error_svd = cross_validator(matrix_factorization_svd, train_df2, 2)

#%%
error_svd = cross_validator(matrix_factorization_svd, train_df, 5)

#%%
prediction = matrix_factorization_svd(train_df, test_df)
create_submission_file(prediction, output_name="submission_svd.csv")

#%%
output = matrix_factorization_svd_normalized(train_df2, train_df2)

#%%
error_svd_rescaled = cross_validator(matrix_factorization_svd_normalized, train_df2, 2)

#%%
error_svd_rescaled = cross_validator(matrix_factorization_svd_normalized, train_df, 5)

#%% Optimizing the parameters
lr_list = [0.01, 0.005, 0.001, 0.0005]
reg_list = [0.1, 0.01, 0.005, 0.001, 0.0005]
error_list = []
param_list = []
for lr in lr_list:
    for reg in reg_list:
         print("lr: {} reg: {}".format(lr,reg))
         error = cross_validator_param_opt(matrix_factorization_svd, train_df, 5, lr_all=lr, reg_all=reg)
         error_list.append(error)
         param_list.append(set([lr,reg]))

error_list.index(np.min(error_list))
param_list[error_list.index(np.min(error_list))]

