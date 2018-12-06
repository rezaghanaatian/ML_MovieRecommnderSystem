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
        self.rank = args.rank
        self. lambda_ = args.lambda_
        self.iterations = args.iterations
        self.spark_context = args.spark_context
        
        
    def predict(self, train, test):
        
        output = test.copy()

        # calculate the global mean
        global_mean_train = train.Prediction.mean()

        output.Prediction = global_mean_train

        def round_pred(row):
            return round(row.Prediction)

        output['Prediction'] = output.apply(round_pred, axis=1)
        output['Rating'] = output['Prediction']
        print(output[:20])
        
        return output
    

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
        self.rank = args.rank
        self. lambda_ = args.lambda_
        self.iterations = args.iterations
        self.spark_context = args.spark_context
        
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
        self.rank = args.rank
        self. lambda_ = args.lambda_
        self.iterations = args.iterations
        self.spark_context = args.spark_context
        
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
        output ['Prediction'] = output['Rating']
                
        return output
    
    
class SGD(optimizer):
    
    def __init__(self, args):
        self.args = args
        
        
    def predict(self, train, test):
        output = test.copy()
        
        #TO DO 
        raise NotImplementedError('This optimizer should be implemented!')

        return output 

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
    
    