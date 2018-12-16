# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 16:48:24 2018

@author: Reza Ghanaatian
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp

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
def compute_error_sp(data, user_features, movie_features, nz):
    """compute the loss (RMSE) of the prediction of nonzero elements.
        The inputs are sparse matrices
    """
    mse = 0
    for row, col in nz:
        movie = movie_features[row,:]
        user = user_features[col,:]
        #mse += (data[row, col] - np.round(np.dot(movie,user.T))) ** 2
        mse += (data[row, col] - np.dot(movie,user.T)) ** 2
    return np.sqrt(1.0 * mse / len(nz))

#%%
def matrix_factorization_sgd(train, test, **kwargs):
    """
    matrix factorization using SGD from surprise library with normalization
    
    input
        train:      pandas dataframe
        test:       pandas dataframe
        **kwargs:   arbitrary keyword argument
        
    output
        pandas dataframe: prediction
    """
   
    # hyper parameter
    gamma = kwargs['gamma']#0.01
    nb_epochs = 20
    nb_latent = kwargs['nb_latent']#20
    lambda_user = 0.1
    lambda_movie = 0.7
    
    
    # init 
    D = max(np.max(train.Movie),np.max(test.Movie))
    N = max(np.max(train.User),np.max(test.User))
    K = nb_latent 
    np.random.seed(988)
    movie_features = np.random.rand(D,K)
    user_features = np.random.rand(N,K)
    
    # convert to scipy.sparse matrices
    train_sp = df_to_sp(train)
    
    
    # find the non-zero indices
    nz_row, nz_col = train_sp.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    
    
    #err_tr = compute_error_sp(train_sp, user_features, movie_features, nz_train)
    #print("iter: {}, RMSE on train set: {}...".format(0, err_tr))
    
    # the gradient loop
    for it in range(nb_epochs):
        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        
        # decrease step size
        gamma /= 1.2
    
        for d, n in nz_train:
            # matrix factorization.
            err = train_sp[d, n] - np.dot(movie_features[d, :], user_features[n, :].T)
            grad_movie = -err * user_features[n, :] + lambda_movie * movie_features[d, :]
            grad_user = -err * movie_features[d, :] + lambda_user * user_features[n, :]
    
            movie_features[d, :] -= gamma * grad_movie
            user_features[n, :] -= gamma * grad_user
            
        #err_tr = compute_error_sp(train_sp, user_features, movie_features, nz_train)
        #print("iter: {}, RMSE on train set: {}...".format(it+1, err_tr))
            
    # do the prediction and fill the test set
    test_sp = df_to_sp(test)
    nz_row, nz_col = test_sp.nonzero()
    nz_test = list(zip(nz_row, nz_col))
    X_pred = np.round(np.dot(movie_features, user_features.T))
    for row, col in nz_test:
        val = X_pred[row, col]
        if val > 5:
            pred= 5
        elif val < 1:
            pred = 1
        else:
            pred = val
        test_sp[row, col] = pred
    
    test_pred = sp_to_df(test_sp)
    
    err_ts = compute_error2(test, test_pred)
    print("RMSE on test set: {}.".format(err_ts))
   
#    test_output = test.copy()
#    X_pred = np.round(np.dot(movie_features, user_features.T))
#    for i, row in test_output.iterrows():
#        test_output.Prediction[i] = X_pred[test_output.Movie[i]-1,test_output.User[i]-1]
#    err_ts = compute_error(test, test_output)
#    print("RMSE on test set: {}.".format(err_ts))

    return test_pred

#%% 
"""
def matrix_factorization_sgd_rescaled(train, test):
    
     # Normalize the train data
    def dict_mean_user(train):
        train=train.astype(int)
        global_mean = train.groupby('User').mean().Prediction.mean()
        return dict(train.groupby('User').mean().Prediction - global_mean)

    train_scaled = pd.DataFrame.copy(train)
    #train_scaled['Prediction'] = train_scaled['Prediction'].astype('float')
    #train_scaled.dtypes
    train_scaled['Prediction'] = train.apply(lambda x: x['Prediction'] - dict_mean_user(train)[x['User']],axis=1)
    
    #do the prediction using the normalized train data
    test_pred = matrix_factorization_sgd(train_scaled, test)
    
    #rescale the prediction to recover the actual mean
    test_pred_rescaled = pd.DataFrame.copy(test_pred)
    test_pred_rescaled['Prediction'] = test_pred.apply(lambda x: x['Prediction'] + dict_mean_user(test_pred)[x['User']],axis=1)
    
    err_ts = compute_error2(test, test_pred_rescaled)
    print("RMSE on test set after rescaling: {}.".format(err_ts))
    
    return test_pred_rescaled
 """   
#%%
def matrix_factorization_sgd_normalized(train, test, **kwargs):
    """
    matrix factorization using SGD with normalization
    
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
    test_pred_normalized = matrix_factorization_sgd(train_normalized, test)
    
    #rescale the prediction to recover the actual mean
    test_pred = recover_deviation(test_pred_normalized)
       
    err_ts = compute_error2(test, test_pred)
    print("RMSE on test set after rescaling: {}.".format(err_ts))
    
    return test_pred    
#%%
train_df2 = train_df.head(1000)
train_df3= train_df.head(100000);

#%%
output = matrix_factorization_sgd(train_df2, train_df2)

#%%
error_sgd = cross_validator(matrix_factorization_sgd, train_df2, 2)

#%%
error_sgd = cross_validator(matrix_factorization_sgd, train_df, 5)


prediction = matrix_factorization_sgd(train_df, test_df)
create_submission_file(prediction, output_name="submission_sgd.csv")

#%%
output = matrix_factorization_sgd_normalized(train_df2, train_df2)

#%%
error_sgd_rescaled = cross_validator(matrix_factorization_sgd_normalized, train_df2, 2)

#%%
error_sgd_rescaled = cross_validator(matrix_factorization_sgd_normalized, train_df, 5)

#%% Optimizing the parameters
gamma_list = [0.02, 0.01, 0.005]
nb_latent_list = [10, 20, 30]
error_list = []
param_list = []
for gm in gamma_list:
    for fac in nb_latent_list:
         print("gamma: {} factors: {}".format(gm,fac))
         error = cross_validator_param_opt(matrix_factorization_sgd, train_df, 5, gamma=gm, nb_latent=fac)
         error_list.append(error)
         param_list.append(set([gm,fac]))

error_list.index(np.min(error_list))
param_list[error_list.index(np.min(error_list))]
