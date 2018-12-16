#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Author: Zahra Farsijani
#
# Distributed under terms of the MIT license.

"""
main_cv.py

run command "spark-submit run.py" to launch it
Cross-validate the models and test the blending
WARNING! Takes a LOT of time!
"""

# import models
from optimizers import createOptimizer
from blend_cv import * 
from helpers_v2 import *
from pyspark import SparkContext, SparkConf
import scipy.optimize as sco
import argparse 
import time


def main(args):
    start = time.time()
    print("[LOG] Blender cross-validation started")
  

    # configure and start spark
    conf = (SparkConf()
            .setMaster("local")
            .setAppName("My app")
            .set("spark.executor.memory", "1g")
            )
    sc = SparkContext(conf=conf)
    args.spark_context = sc
    
    print("[LOG] Start blending for recommender system")

    # test if spark works
    if sc is not None:
        print("[INFO] Spark successfully initiated")
    else:
        print("[ERROR] Problem with spark, check your configuration")
        exit()

    # hide spark log information
    sc.setLogLevel("ERROR")

    #Path to Datasets
    path_dataset = "data/data_train.csv"
    path_test_dataset = "data/sample_submission.csv"
     
    #Load Datasets    
    print("[LOG] Loading train set... \n")
    train_df = load_dataset(path_dataset)
    test_df  = load_dataset(path_test_dataset)
    #train_df = train_df[:20]
    #test_df  = test_df[:20]
    
    # Add 'Rating' column that is the copy of 'Prediciotn' - this column is needed in some models such as ALS...
    train_df['Rating'] = train_df['Prediction']
    test_df ['Rating'] = test_df['Prediction']

    # Get the models (kes: models names, values: model parameters)
    all_models = ['global_mean', 'movie_mean', 'user_mean', 'als', 'als_normalized']
    models = get_models(args)  

    print("[LOG] Cross-validator started.")

    # Createthe CrossValidator object and start the cross-validation
    cv = BlendCrossValidator()
    cv.new_validator(train_df, 4, True)
    
#    # Split the train set in 5 folds
#    cv.shuffle_indices_and_store(train, 5)

    ############## PREDICT AND STORE THE MODELS ################
    for model_name in all_models:
        args.optimizer = model_name
        model = createOptimizer(args)
        print("[LOG] Predicting and storing model " + model_name) # 
        tt = time.time()
        cv.k_fold_predictions_and_store(train_df, model, model_name, True, args)#models[model_name]['params'])
        print("[LOG] Completed in %s\n" % (time_str(time.time() - tt)))

    # Delete the cross-validator object to free memory
    del cv

    print("[LOG] Reloading cross-validator object...")
    # Reload the CrossValidator class
    cv = BlendCrossValidator()

    # Load the indices
    cv.load_indices()

    # Load the ground truth
    cv.define_ground_truth(test_df)

    model_names = list(models.keys())

    # Load model predictions
    cv.load_predictions(model_names)

    # Define first vector for Blending
    x0 = 1 / len(model_names) * np.ones(len(model_names))

    # carry out the Optimization for the blending
    print("[LOG] Optimization for the blending started..." + '\n' + "Please be patient!")
    
    res = sco.minimize(eval_, x0, method='SLSQP', args=(cv, models), options={'maxiter': 1000, 'disp': True})

    # Create best dictionnary
    best_dict = {}
    for idx, key in enumerate(models.keys()):
        best_dict[key] = res.x[idx]

    # Test the blending
    test_blending(cv, best_dict)

    print("[LOG] Blending was done in: %s" % (time_str(time.time() - start)))

    print("==============")
    print("[LOG] FINISHED!")
    print("==============")


def eval_(x, cv, models):
    """ Evaluate the RMSE of the blending """
    dict_try = {}
    for idx, key in enumerate(models.keys()):
        dict_try[key] = x[idx]

    return cv.evaluate_blending(dict_try)


def test_blending(cv, best_dict):
    """ Evaluate the RMSE of each model and a specific blending """
    cv.evaluation_all_models()

    print()
    rmse = cv.evaluate_blending(best_dict)
    print("Best blending: %s" % best_dict)
    print("RMSE best blending: %.5f" % rmse)


def get_models(args):
    
    """
    Returns the a list of models along with their parameters
    
    Args:
        args: as entered by the user or the default values 
        
    Retursn:
        models(python dict): Dictionary containing the model names as keys and parameters as values
    """
    
    models = {}
    # name of optimization all models implemented in this project
    all_models = ['global_mean', 'movie_mean', 'user_mean', 'als', 'als_normalized']
    
    for model in all_models:
        args.optimizer = model
        optimizer_obj = createOptimizer(args)
        models[model] = optimizer_obj.get_params()
        
        
    return models


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument('--optimizer', type=str, default="blended",  help='Only \'blended\' available!')
    parser.add_argument('--rank', type=int, default=8, help='rank(i.e. number of latent factors)')
    parser.add_argument('--lambda_', type=float, default=0.07, help='lambda in ALS optimizer')
    parser.add_argument('--iterations', type=int, default=24, help='Number of iterations')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds used in cross-validation')
    parser.add_argument('--output_folder', type=str, default="output/", help='Output folder address')
    parser.add_argument('--file_name', type=str, default="sub", help='Name of submission file w/o .csv')
  
    args = parser.parse_args()
    main(args)