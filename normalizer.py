#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Author: Zahra Farsijani
# Version: 1.0
#


"""
Normalize the data besd on mean, variation or deviation.

USAGE:
    normalizer = Normalizer(df)
    normalized_train_set = normalized.normalize_deviation()

        (do the ML training which return `predicted_set` )

    test_set = normalizer.recover_deviation(predicted_set)

"""

import pandas as pd
from pyspark.mllib.recommendation import Rating


def dict_mean_user(df):
    """ dictionary: key=UserID & value=User Mean """
    return dict(df.groupby('User').mean().Rating)


def dict_var_user(df):
    """ dictionary: key=UserID & value=User Mean """
    return dict(df.groupby('User').var().Rating)


def dict_dev_user(df):
    """ dictionary: key=UserID & value=User Mean  """
    global_mean = df.groupby('User').mean().Rating.mean()
    return dict(df.groupby('User').mean().Rating - global_mean)

class Normalizer(object):
    """
    This class normalizes the dataframe

    Its methods provide dataframe normalization as well as recovery to get the right predictions
    from the predictions obtained from the normalized version of the dataframe
    """
        
    def __init__(self, df):
        """
        Computes mean, variance and deviation of dataframe.

        Params:
            data_frame: Pandas dataframe to normalize
        """
        self.df = df
        self.variances = dict_var_user(df)
        self.means = dict_mean_user(df)
        self.deviation = dict_dev_user(df)
        
    def normalize_gaussian(self, df):
        """ Return gaussian normalized dataframe - both mean and variance
        """
        gnorm_df = pd.DataFrame.copy(self.df)
        gnorm_df['Prediction'] = self.df.apply(
            lambda x: (x['Prediction'] - self.means[x['User']]) / self.variances[x['User']],
            axis=1)

        return gnorm_df

    def recover(self, df):
        """ Recover from 'normalized' table
        """
        recovered_df = pd.DataFrame.copy(df)
        recovered_df['Prediction'] = df.apply(
            lambda x: (x['Prediction'] * self.variances[x['User']]) + self.means[x['User']],
            axis=1)

        return recovered_df

    def normalize_mean(self,df):
        """ Set all mean to 0
        """
        norm_df = pd.DataFrame.copy(self.df)
        norm_df['Prediction'] = self.df.apply(
            lambda x: x['Prediction'] - self.means[x['User']],
            axis=1)

        return norm_df

    def recover_mean(self, df):
        """ Recover from 'normalized_only_mean' table
        """
        recovered_df = pd.DataFrame.copy(df)
        recovered_df['Prediction'] = df.apply(
            lambda x: x['Prediction'] + self.means[x['User']],
            axis=1)

        return recovered_df

    def normalize_deviation(self,df):
        """ Set all the mean to overall mean (BEST METHOD)
        """
        norm_df = pd.DataFrame.copy(self.df)
        norm_df['Prediction'] = self.df.apply(
            lambda x: x['Prediction'] - self.deviation[x['User']],
            axis=1)

        return norm_df

    def recover_deviation(self, df):
        """ Recover from 'normalized_deviation' table
        """
        recovered_df = pd.DataFrame.copy(df)
        recovered_df['Prediction'] = data_frame.apply(
            lambda x: x['Prediction'] + self.deviation[x['User']],
            axis=1)

        return recovered_df
        
        
        