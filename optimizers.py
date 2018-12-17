#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Author: Zahra Farsijani
#
# This File contains the classes that implement optimization models

from abc import abstractmethod
import numpy as np
from normalizer import Normalizer
from pyspark.mllib.recommendation import ALS
from pyspark.sql import SQLContext
import os
from helpers_v2 import *
from helpers import df_to_sp, sp_to_df

import scipy.sparse as sp
from surprise.prediction_algorithms import SVD
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split

class optimizer(object):
    @abstractmethod
    def compute_prediction(self, prediction):
        raise NotImplementedError('compute_prediction method is not implemented.')
        
        
class GlobalMean(optimizer):
    """
        Define the global mean model for recommendation.

        Args:
            train (Pandas Dataframe) : train dataset
            test (Pandas Dataframe): test dataset

        Returns:
            output (Pandas Dataframe): test dataset with updated predictions calculated with global mean
    """
    
    def __init__(self, args):
        
        self.iterations = args.iterations
        
        
        
    def predict(self, train, test):
        
        output = test.copy()

        # calculate the global mean
        global_mean_train = train.Prediction.mean()

        output.Prediction = global_mean_train

        def round_pred(row):
            return round(row.Prediction)

        output['Prediction'] = output.apply(round_pred, axis=1)
        output['Rating'] = output['Prediction']    
        return output
    
    def get_params(self):
        params = []
        params = [self.iterations]
        return params 
    
    def set_params(self,params):
        
        self.iterations = args.iterations
    
         

class UserMean(optimizer):
    """
       Define the user mean model for recommendation. recommends based on user's average rating.

        Args:
            train (Pandas Dataframe) : train dataset
            test (Pandas Dataframe): test dataset

        Returns:
            output(Pandas Dataframe): test dataset with updated predictions calculated with global mean
    """
    def __init__(self, args):
        
        self.iterations = args.iterations
        self.rank = args.rank
        self. lambda_ = args.lambda_
        self.spark_context = args.spark_context
        self.args = args
        
    def predict(self, train, test):    
        
        output = test.copy()
        train.User = train.User.astype(int)
        train.Prediction = train.Prediction.astype(int)

        # calculate the mean for each user
        mean_pred_by_user = train.groupby('User')[['Prediction']].mean()

        def assign_mean(row):
            return mean_pred_by_user[mean_pred_by_user.index == row.User].iloc[0].Prediction

        def round_pred(row):
            return round(row.Prediction)

        output['Prediction'] = output.apply(assign_mean, axis=1)
        output['Prediction'] = output.apply(round_pred, axis=1)
        output['Rating'] = output['Prediction']
        return output
    
    def get_params(self):
        params = []
        params = [self.iterations]
        return params 
    
    def set_params(self,params):
        
        self.iterations = args.iterations
    
        
        
    
class MovieMean(optimizer):
    """
        Define the user mean model for recommendation. recommends based on movie's average rating.

        Args:
            train (Pandas Dataframe) : train dataset
            test (Pandas Dataframe): test dataset

        Returns:
            (Pandas Dataframe): test dataset with updated predictions calculated with movie mean

    """
    def __init__(self, args):
        self.iterations = args.iterations
        
    def predict(self, train, test):

        output = test.copy()
        train.Movie = train.Movie.astype(int)
        train.Prediction = train.Prediction.astype(int)

        # calculate the mean for each movie
        mean_pred_by_movie = train.groupby('Movie')[['Prediction']].mean()

        def assign_mean(row):
            return mean_pred_by_movie[mean_pred_by_movie.index == row.Movie].iloc[0].Prediction

        def round_pred(row):
            return round(row.Prediction)

        output['Prediction'] = output.apply(assign_mean, axis=1)
        output['Prediction'] = output.apply(round_pred, axis=1)
        output['Rating'] = output['Prediction']
        return output

    def get_params(self):
        params = []
        params = [self.iterations]
        return params 
    
    def set_params(self,params):
        
        self.iterations = args.iterations
    
    
class ALSOptimizer(optimizer):
    
    """
    ALS with PySpark.

    Compute the predictions on a test set after training on a train set using the ALS method from PySpark.

    Args:
        train (pd.DataFrame): train set
        test (pd.DataFrame): test set
        args: Arbitrary keyword arguments. Passed to ALS.train() (Except for the spark_context)
            spark_context: SparkContext passed from the main program. (Useful when using Jupyter)
            rank (int): number of latent factors to use 
            lambda_ (float): regularization parameter in ALS
            iterations (int): number of iterations of ALS to run. 
            (ALS typically converges to a reasonable solution in 20 iterations or less)

    Returns:
        Pandas DataFrame: predictions, sorted as (Movie, User)
    """
    def __init__(self, args):
        self.rank = args.rank
        self. lambda_ = args.lambda_
        self.iterations = args.iterations
        self.spark_context = args.spark_context
        self.args = args

    
    def predict(self, train, test):

        # Delete folders that cause trouble while running the code
        os.system('rm -rf metastore_db')
        os.system('rm -rf __pycache__')
        
       
        train.Movie = train.Movie.astype(int)
        train.Prediction = train.Prediction.astype(int)
        train.Rating = train.Rating.astype(int)
        
        # Prepare the dataFrame to be used in ALS object instantiation with headings
        # ['index','Prediction',User','Movie','Rating']
        train = train.drop(['Prediction'], axis=1)
        test = test.drop(['Prediction'], axis=1)
        
        
        output = test.copy()
        
        # Convert pd.DataFrame to Spark.rdd 
        sqlContext = SQLContext(self.spark_context)

        train_sql = sqlContext.createDataFrame(train).rdd
        test_sql = sqlContext.createDataFrame(test).rdd

        # Train the model
        print("[LOG] ALS training started; this may take a while!")
        model = ALS.train(train_sql, rank=self.rank, lambda_=self.lambda_, iterations=self.iterations)

        # Predict
        to_predict = test_sql.map(lambda p: (p[0], p[1]))
        predictions = model.predictAll(to_predict).map(lambda r: ((r[0], r[1]), r[2]))

        # Convert Spark.rdd back to pd.DataFrame
        output = predictions.toDF().toPandas()
        

        # Postprocesse  database
        output['User'] = output['_1'].apply(lambda x: x['_1'])
        output['Movie'] = output['_1'].apply(lambda x: x['_2'])
        output['Rating'] = output['_2']
        output = output.drop(['_1', '_2'], axis=1)
        output['Prediction'] = output['Rating']
        output = output.sort_values(by=['Movie', 'User'])
        output.index = range(len(output))
      
        
        def round_pred(row):
            return round(row.Prediction)
        
        output['Prediction'] = output.apply(round_pred, axis=1)
        output['Rating'] = output['Prediction']
       
        return output
        
    def get_params(self):
        params = []
        params = [self.rank ,\
                  self. lambda_,\
                  self.iterations,\
                  self.spark_context]
        return params 
    
    def set_params(self,params):
        
        self.rank = args.rank
        self. lambda_ = args.lambda_
        self.iterations = args.iterations
        self.args = args
                        
        


class ALSNormalizedOptimizer(optimizer):
    
    """
    ALS with normalized user ratings to compensate for user 'mood'.

    First, normalize the user so that they all have the same average of ratings. 
    Then, predict with explicit preferences given by the user to the item, 
    for example, users giving ratings to movies.
    Finally, recover the deviation of each user.

    Input:
        train (pandas.DataFrame): train set
        test (pandas.DataFrame): test set
            spark_context: SparkContext passed from the main program. (Useful when using Jupyter)
            rank (int): number of latent factors to use 
            lambda_ (float): regularization parameter in ALS
            iterations (int): number of iterations of ALS to run. 
            (ALS typically converges to a reasonable solution in 20 iterations or less)
            

    Output:
        pandas.DataFrame: predictions, sorted as ['index', 'Prediction', 'User', 'Movie']
    """
    def __init__(self, args):
        self.rank = args.rank
        self. lambda_ = args.lambda_
        self.iterations = args.iterations
        self.spark_context = args.spark_context
        self.args = args

    
    def predict(self, train, test):
        
        # Instantiate the Normalizer class
        normalizer = Normalizer(train)

        # Normalize the train data - set all the mean to overall mean
        df_train_normalized = normalizer.normalize_deviation()
       

        # Predict using the normalized trained data
        alsModel = ALSOptimizer(self.args)
        prediction_normalized = alsModel.predict(df_train_normalized, test)
        

        # Recover the prediction to recover the deviations
        output = normalizer.recover_deviation(prediction_normalized)
       
        
        def round_pred(row):
            return round(row.Prediction)
        
        output['Prediction'] = output.apply(round_pred, axis=1)
        output ['Rating'] = output['Prediction']
                
        return output
    
    def get_params(self):
        params = []
        params = [self.rank ,\
                  self. lambda_,\
                  self.iterations,\
                  self.spark_context]
        return params 
    
    def set_params(self,params):
        
        self.rank = args.rank
        self. lambda_ = args.lambda_
        self.iterations = args.iterations
        self.args = args
         
    
class SGD(optimizer):
    """
    matrix factorization using SGD 
    
    input
        train:      pandas dataframe
        test:       pandas dataframe
        **kwargs:   arbitrary keyword argument
        
    output
        pandas dataframe: prediction
    """
    def __init__(self, args):
        self.args = args
        
        
    def predict(self, train, test):

        # hyper parameter
        gamma = 0.02
        nb_epochs = 20
        nb_latent = 30
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
        return test_pred


    def get_params(self):
        params = []
        params = [self.rank ,\
                  self. lambda_,\
                  self.iterations,\
                  self.spark_context]
        return params 
    
    def set_params(self,params):
        
        self.rank = args.rank
        self. lambda_ = args.lambda_
        self.iterations = args.iterations
        self.args = args
 

class SGDNormalized(optimizer):
    """
    matrix factorization using SGD with normalization
    
    input
        train:      pandas dataframe
        test:       pandas dataframe
        **kwargs:   arbitrary keyword argument
        
    output
        pandas dataframe: prediction
    """    
    def __init__(self, args):
        self.args = args
        
        
    def predict(self, train, test):
        
        train['Rating'] = train['Prediction']
        #train = train.rename(index=str, columns={"Prediction" : "Rating"})
        
        # instance normalizer
        normalizer = Normalizer(train)
        
        # do the normalization
        train_normalized = normalizer.normalize_deviation()
        train['Prediction'] = train['Rating']
        train_normalized = train_normalized.drop(labels=["Rating"], axis=1)
        
        # Predict using the normalized trained data
        sgdmodel = SGD('')
        test_pred_normalized = sgdmodel.predict(train_normalized, test)
        
        #rescale the prediction to recover the actual mean
        test_pred_normalized['Rating'] = test_pred_normalized['Prediction']
        test_pred = normalizer.recover_deviation(test_pred_normalized)
        test_pred['Prediction'] = test_pred['Rating']
        test_pred = test_pred.drop(labels=["Rating"], axis=1)
        
        def round_pred(row):
            return round(row.Prediction)
        
        test_pred['Prediction'] = test_pred.apply(round_pred, axis=1)
        
        return test_pred
       


        
def createOptimizer(args):
    
    if args.optimizer == "global_mean":
        return GlobalMean(args)
    elif args.optimizer == "user_mean":
        return UserMean(args)
    elif args.optimizer == "movie_mean":
        return MovieMean(args)
    elif args.optimizer == "als":
        return ALSOptimizer(args)
    elif args.optimizer == "als_normalized":
        return ALSNormalizedOptimizer(args)
    elif args.optimizer == "sgd":
        return SGD(args)
    elif args.optimizer == "sgd_normalized":
        return SGDNormalized(args)

