#import models
from optimizers import createOptimizer
from helpers_v2 import *
from pyspark import SparkContext, SparkConf
import argparse 
import time


""" load dataset """

def main(args):
   
    start = time.time()
    print("============")
    print("[LOG] START")
    print("============")

    print("[LOG] Starting Spark...")

    # configure and start spark
    conf = (SparkConf()
            .setMaster("local")
            .setAppName("My app")
            .set("spark.executor.memory", "1g")
            )
    sc = SparkContext(conf=conf)
    args.spark_context = sc

    # test if spark works
    if sc is not None:
        print("[LOG] Spark successfully initiated!")
    else:
        print("[ERROR] Problem with spark, check your configuration")
        exit()

    # Hide spark log information
    sc.setLogLevel("ERROR")
    
    #Load Datasets
    path_dataset = "data/data_train.csv"
    path_test_dataset = "data/sample_submission.csv"

    train_df = load_dataset(path_dataset)
    test_df = load_dataset(path_test_dataset)
   
    # Add 'Rating' column that is the copy of 'Prediciotn' - this column is needed in some models such as ALS...
    train_df['Rating'] = train_df['Prediction']
    test_df ['Rating'] = test_df['Prediction']
    
    # List of all models implemented in optimizers.py
    if args.optimizer=='blended' or args.optimizer=='all':
        all_models = ['als', 'global_mean', 'user_mean', 'movie_mean']
    else:
        all_models = [args.optimizer]
    
    print("[LOG] Recommender System modeling started")
    models = {}
    error = {}
    
    # Use different optimization models for prediction - create optimizer dictionary object 
    for m in all_models:
        
        args.optimizer = m
        models[m] = createOptimizer(args)
      
        print("[LOG] Preparing model " + '\"'+ args.optimizer + '\"...')
        tt = time.time()

        # Predict the ratings...
        predictions = models[m].predict(train_df, test_df)
        print("[LOG] Completed in %s\n" % (time_str(time.time() - tt)))

        # Cross-validation on train set and calculating RMSE
        tt_cv = time.time()
        n_folds = args.n_folds
   
        print("[LOG] Now running cross-validation...")
        error[m] = cross_validator(models[m], train_df, n_folds)
        print("[LOG] Cross-validation done!")
        print("[LOG] Cross-validation was completed in %s\n" % (time_str(time.time() - tt_cv)))
        print("[LOG] Test RMSE of the baseline using {}: {}".format(args.optimizer,error[m]))

        # If model 'blended' is chosen, do not save models in seperate files!
        if args.optimizer == 'blended':
            break
        
        #Create submission file
        print("[LOG] Preparing submission file ...")
        file_name = args.file_name + '_' + args.optimizer + '.csv'
        output_folder = args.output_folder
        sub_df = create_submission_file(predictions, file_name, output_folder)
        print("[LOG] Successfully wrote to file "+ file_name)
        
    """ Combine different models to better predict - upon user's request"""
    if args.optimizer == 'blended':
        
        n_models = len(all_models)
        
        # Initialize weights needed for combining models
        weight = {
        'global_mean': float(1)/n_models, #1.7756776068889906,
        'global_median': 1.8469393889491512,
        'user_mean': float(1)/n_models, #-3.6424669808916055,
        'user_median': 0.0051375146192670111,
        'movie_mean': float(1)/n_models, #-0.83307991660204828,
        'movie_mean_rescaled': -0.95695560022481185,
        'movie_median': -0.93869701618369406,
        'movie_median_rescaled': -0.91347225736204185,
        'movie_mean_deviation_user': 1.0442870681129861,
        'movie_mean_deviation_user_rescaled': 0.92108939957699987,
        'movie_median_deviation_user': 0.93849170091288214,
        'movie_median_deviation_user_rescaled': 0.96461941548011165,
        'mf_rr': 0.032225151029461573,
        'mf_rr_rescaled': 0.035378890871598068,
        'mf_sgd': -0.78708629851314926,
        'mf_sgd_rescaled': 0.27624842029358976,
        'als': 0.30659162734621315,
        'als_normalized': float(1)/n_models, #0.31745406600610854,
        'pyfm': 0.15296423817447555,
        'pyfm_rescaled': -0.021626658245201873,
        'baseline': -0.70720719475460081,
        'baseline_rescaled': -0.56908887025195931,
        'slope_one': -0.023119356625828508,
        'slope_one_rescaled': 0.43863736787065016,
        'svd': 0.67558779271650848,
        'svd_rescaled': -0.0049814548552847716,
        'knn_ib': -0.095005112653966148,
        'knn_ib_rescaled': 0.34178799145510136,
        'knn_ub': 0.21758562399412981,
        'knn_ub_rescaled': 0.12803210410741006
        }
        
        
        print("[LOG] Now blending models...")
        blend = blender(models, weights)
        
        # Cross-validation on train set and calculating RMSE
        tt_cv = time.time()
        n_folds = args.n_folds
        print("[LOG] Now running cross-validation on blend...")
        error_b = cross_validator(blend, train_df, n_folds)
        print("[LOG] Blend cross-validation done!")
        print("[LOG] Blend cross-validation was completed in %s\n" % (time_str(time.time() - tt_cv)))
        print("[LOG] Test RMSE of the blend using {}: {}".format(args.optimizer,error_b))
        
        #Create submission file
        print("[LOG] Preparing submission file ...")
        file_name = args.file_name + '_' + args.optimizer + '.csv'
        output_folder = args.output_folder
        sub_df_b = create_submission_file(blend, file_name, output_folder)
        print("[LOG] Successfully wrote to file "+ file_name)

    
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument('--optimizer', type=str, default="als",  help='Name of the optimizer: als,\
       als_normalized, global_mean, user_mean, movie_mean, sgd, blended, all')
    parser.add_argument('--rank', type=int, default=8, help='rank(i.e. number of latent factors)')
    parser.add_argument('--lambda_', type=float, default=0.081, help='lambda in ALS optimizer')
    parser.add_argument('--iterations', type=int, default=24, help='Number of iterations')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds used in cross-validation')
    parser.add_argument('--output_folder', type=str, default="output/", help='Output folder address')
    parser.add_argument('--file_name', type=str, default="sub", help='Name of submission file w/o .csv')
  
    args = parser.parse_args()
    main(args)
