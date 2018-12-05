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

    # hide spark log information
    sc.setLogLevel("ERROR")
    
    
    print("[LOG] Recommender System modeling started")
   

    #Load Datasets
    path_dataset = "data/data_train.csv"
    path_test_dataset = "data/sample_submission.csv"

    train_df = load_dataset(path_dataset)
    test_df = load_dataset(path_test_dataset)
    
    # Add 'Rating' column that is the copy of 'Prediciotn' - this column is needed in some models such as ALS...
    train_df['Rating'] = train_df['Prediction']
    test_df ['Rating'] = test_df['Prediction']
    
    # Use different optimization models for prediction - create optimizer object
    model = createOptimizer(args)
    
    print("[LOG] Preparing model " + '\"'+ args.optimizer + '\"...')
    tt = time.time()
    
    # Predict 
    predictions = model.predict(train_df, test_df)
    print("[LOG] Completed in %s\n" % (time_str(time.time() - tt)))
    
    
    # Calculate rmse using cross-validation on train set
    tt_cv = time.time()
    n_folds = args.n_folds
    print("[LOG] Now running cross-validation...")
    error = cross_validator(model, train_df, n_folds)
    print("[LOG] Cross-validation done!")
    print("[LOG] Cross-validation completed in %s\n" % (time_str(time.time() - tt_cv)))
    print("[LOG] Test RMSE of the baseline using {}: {}".format(args.optimizer,error))
    
    


    """ Combine different models to better predict """
    #### TO DO

    
    #Create submission file
    print("[LOG] Preparing submission file ...")
    file_name = args.file_name
    output_folder = args.output_folder
    sub_df = create_submission_file(predictions, file_name, output_folder)
    print("[LOG] Successfully wrote to file "+ file_name)

    
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument('--optimizer', type=str, default="als_normalized",  help='Name of the optimizer: als,\
       als_normalized, global_mean, user_mean, movie_mean, sgd, blended')
    parser.add_argument('--rank', type=int, default=8, help='rank(i.e. number of latent factors)')
    parser.add_argument('--lambda_', type=float, default=0.081, help='lambda in ALS optimizer')
    parser.add_argument('--iterations', type=int, default=24, help='Number of iterations')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds used in cross-validation')
    parser.add_argument('--output_folder', type=str, default="output/", help='Output folder address')
    parser.add_argument('--file_name', type=str, default="submission.csv", help='Name of submission file')
  
    args = parser.parse_args()
    main(args)
