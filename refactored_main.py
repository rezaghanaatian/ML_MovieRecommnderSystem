import argparse
import time

from helpers import load_dataset, create_submission_file
from models_blender import Blender
from models_knn import SurpriseKNN
from models_mean import GlobalMean, UserMean, MovieMean
from models_median import GlobalMedian, MovieMedian, UserMedian
from models_als import ALSOptimizer
from models_sgd import SGDOptimizer
from models_svd import SurpriseSVD


""" load dataset """


def main(args):
    start = time.time()
    print("In the name of God\n============\n[LOG] START\n============")

    # Load Datasets
    path_dataset = "data/data_train.csv"
    path_test_dataset = "data/sample_submission.csv"

    train_df = load_dataset(path_dataset)
    test_df = load_dataset(path_test_dataset)
    train_def = train_df.head(20)
    test_df = test_df.head(20)

    # Initialize models here:
    prediction_models = []
    #knn = SurpriseKNN(k=60, user_based=False)
    #prediction_models.append(knn)
    #global_mean = GlobalMean()
    #prediction_models.append(global_mean)
    #user_mean = UserMean()
    #prediction_models.append(user_mean)
    #movie_mean = MovieMean()
    #prediction_models.append(movie_mean)
    # global_median = GlobalMedian()
    # prediction_models.append(global_median)
    # movie_median = MovieMedian()
    # prediction_models.append(movie_median)
    # user_median = UserMedian()
    # prediction_models.append(user_median)
    #als = ALSOptimizer()
    #prediction_models.append(als)
    #sgd = SGDOptimizer()
    #prediction_models.append(sgd)
    svd = SurpriseSVD()
    prediction_models.append(sgd)

    # best_weights = Blender.tune_weights(prediction_models, train_df)
    # print(best_weights)

    best_weights = [1] #0.22, 0.22, 0.22]

    blender_model = Blender(models=prediction_models, weights=best_weights)
    blender_model.fit(train_df)
    pred = blender_model.predict(test_df)
    print(pred.head())

    print("============\n[LOG] SAVE RESULT IN CSV FILE\n============")
    create_submission_file(pred, "output.csv", round_predictions=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--sample', type=int, default=0, help='Help text!')
    args = parser.parse_args()
    main(args)
