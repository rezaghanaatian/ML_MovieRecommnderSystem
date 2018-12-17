# import models
from optimizers import createOptimizer
from helpers_v2 import *
from pyspark import SparkContext, SparkConf
import argparse
import time
from optimizers_knn import SurpriseKNN
from refactored.models_mean import GlobalMean
from refactored.prediction_model import PredictionModel

""" load dataset """


def main(args):
    start = time.time()
    print("============")
    print("[LOG] START")
    print("============")


    # TODO: Can we move this part to __init__ of its model?
    # print("[LOG] Starting Spark...")
    #
    # # configure and start spark
    # conf = (SparkConf()
    #         .setMaster("local")
    #         .setAppName("My app")
    #         .set("spark.executor.memory", "1g")
    #         )
    # sc = SparkContext(conf=conf)
    # args.spark_context = sc
    #
    # # test if spark works
    # if sc is not None:
    #     print("[LOG] Spark successfully initiated!")
    # else:
    #     print("[ERROR] Problem with spark, check your configuration")
    #     exit()
    #
    # # Hide spark log information
    # sc.setLogLevel("ERROR")

    # Load Datasets
    path_dataset = "../data/data_train.csv"
    path_test_dataset = "../data/sample_submission.csv"

    train_df = load_dataset(path_dataset)
    test_df = load_dataset(path_test_dataset)

    # Initialize models here:
    prediction_models = []
    prediction_models.append(GlobalMean(train_df))
    # prediction_models.append(SurpriseKNN(train_df, k=50, user_based=False))
    # prediction_models.append(SurpriseKNN(train_df, k=50, user_based=False))
    # TODO add other models

    print("[LOG] Recommendation System modeling started")
    predictions = []

    for model in prediction_models:

        if not issubclass(type(model), PredictionModel):
            continue

        print("[LOG] Preparing model === {0} ===".format(model.get_name()))
        tt = time.time()

        predictions.append(model.predict(test_df))
        print(predictions[0].head(5))
        print("[LOG] Prediction by {0} completed in {1}".format(model.get_name(), time.time() - tt))

    print("[LOG] blending models...")
    # blend = blender(models, weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--optimizer', type=str, default="als", help='Name of the optimizer: als,\
       als_normalized, global_mean, user_mean, movie_mean, sgd, blended, all')
    parser.add_argument('--rank', type=int, default=8, help='rank(i.e. number of latent factors)')
    parser.add_argument('--lambda_', type=float, default=0.081, help='lambda in ALS optimizer')
    parser.add_argument('--iterations', type=int, default=24, help='Number of iterations')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds used in cross-validation')
    parser.add_argument('--output_folder', type=str, default="output/", help='Output folder address')
    parser.add_argument('--file_name', type=str, default="sub", help='Name of submission file w/o .csv')

    args = parser.parse_args()
    main(args)
