# -*- coding: utf-8 -*-
"""
Created on Dec  11  2018

@author: Reza Ghanaatian
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction import DictVectorizer
from pyfm import pylibfm



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
def prepare_data(df):
    """
    Prepare the data for the specific format used by PyFM.

    Args:
        df (pd.DataFrame): Initial DataFrame to transform

    Returns:
        data (array[dict]): Array of dict with user and movie ids
        y (np.array): Ratings give in the initial pd.DataFrame
        users (set): Set of user ids
        movies (set): Set of movie ids

    """
    data = []
    y = list(df.Prediction)
    users = set(df.User.unique())
    movies = set(df.Movie.unique())
    usrs = list(df.User)
    mvies = list(df.Movie)
    for i in range(len(df)):
        y[i] = float(y[i])
        data.append({"user_id": str(usrs[i]), "movie_id": str(mvies[i])})
    return (data, np.array(y), users, movies)


#%%
def factorization_machine_pyfm(train, test, **kwargs):
    """
    factorization machine using pyFM 
    
    input
        train:      pandas dataframe
        test:       pandas dataframe
        **kwargs:   arbitrary keyword argument
        
    output
        pandas dataframe: prediction
    """
    
    # hyper parameters
    n_epochs = 200#kwargs['n_epochs']
    n_factors = 20#kwargs['n_factors'] 
    lr_all = 0.002#kwargs['lr_all']
    task = 'regression'
    learning_schedule = 'optimal'
    
    # prepaper the data to be used in pyFM
    (train_data, train_pred, train_users, train_movies) = prepare_data(train)
    (test_data, test_pred, test_users, test_movies) = prepare_data(test)
    
    v = DictVectorizer()
    x_train = v.fit_transform(train_data)
    x_test = v.transform(test_data)
    
    # training
    fm = pylibfm.FM(num_factors=n_factors, num_iter=n_epochs, task=task, initial_learning_rate=lr_all, learning_rate_schedule=learning_schedule, verbose=False,)
    fm.fit(x_train,train_pred)
    
    # prediction
    preds = fm.predict(x_test)
    vals = preds.copy()
    
    for i in range(len(preds)):
        if preds[i] > 5:
            vals[i] = 5
        elif preds[i] < 1:
            vals[i] = 1
        else:
            vals[i]=np.round(preds[i])
                
                
    test_pred = test.copy()
    test_pred.Prediction = vals    
    err_ts = compute_error2(test, test_pred)
    print("RMSE on test set: {}.".format(err_ts))
       
    return test_pred

    
#%%
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
    
def factorization_machine_pyfm_normalized(train, test, **kwargs):
    """
    factorization machine using pyFM  with normalization
    
    input
        train:      pandas dataframe
        test:       pandas dataframe
        **kwargs:   arbitrary keyword argument
        
    output
        pandas dataframe: prediction
    """

    train_normalized = normalize_deviation(train)
    print ("******************")
    print(train_normalized[:10])

    # Predict using the normalized trained data
    test_pred_normalized = factorization_machine_pyfm(train_normalized, test)
    
    #rescale the prediction to recover the actual mean
    test_pred = recover_deviation(test_pred_normalized)
       
    err_ts = compute_error2(test, test_pred)
    print("RMSE on test set after rescaling: {}.".format(err_ts))
    
    return test_pred    
#%%
train_df2 = train_df.head(1000)
train_df3= train_df.head(10000);

#%%
output = factorization_machine_pyfm(train_df2, train_df2)

#%%
error_svd = cross_validator(factorization_machine_pyfm, train_df3, 2)

#%%
error_svd = cross_validator(factorization_machine_pyfm, train_df, 5)


prediction = factorization_machine_pyfm(train_df, test_df)
create_submission_file(prediction, output_name="submission_pyfm.csv")

#%%
output = factorization_machine_pyfm_normalized(train_df2, train_df2)

#%%
error_svd_rescaled = cross_validator(factorization_machine_pyfm_normalized, train_df3, 2)

#%%
error_svd_rescaled = cross_validator(factorization_machine_pyfm_normalized, train_df, 5)

#%% Optimizing the parameters
n_epochs_list = [30, 100, 200]
n_factors_list = [20, 30, 40]
lr_list = [0.002 0.001 0.002, 0.001, 0.0005]
error_list = []
param_list = []
for ep in n_epochs_list:
    for fac in n_factors_list:
        for lr in lr_list:
            print("epochs: {} factors: {} lr: {}".format(ep,fac,lr))
            error = cross_validator_param_opt(factorization_machine_pyfm, train_df, 5, n_epochs=ep, n_factors=fac, lr_all=lr)
            error_list.append(error)
            param_list.append(set([ep,fac,lr]))

error_list.index(np.min(error_list))
param_list[error_list.index(np.min(error_list))]