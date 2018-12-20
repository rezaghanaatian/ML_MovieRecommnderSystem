#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 21:03:33 2018

@author: ghanaati
"""
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction import DictVectorizer
from pyfm import pylibfm
from helpers import prepare_data

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
    n_epochs = 200
    n_factors = 20
    lr_all = 0.002
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
       
    return test_pred