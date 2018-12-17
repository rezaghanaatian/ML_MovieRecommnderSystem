import argparse
import time

from helpers import load_dataset, create_submission_file
from models_blender import Blender
from models_mean import GlobalMean
from models_median import GlobalMedian

""" load dataset """


def main(args):
    start = time.time()
    print("============\n[LOG] START\n============")

    # Load Datasets
    path_dataset = "data/data_train.csv"
    path_test_dataset = "data/sample_submission.csv"

    train_df = load_dataset(path_dataset)
    test_df = load_dataset(path_test_dataset)

    # Initialize models here:
    prediction_models = []
    global_mean = GlobalMean()
    prediction_models.append(global_mean)
    global_median = GlobalMedian()
    prediction_models.append(global_median)

    best_weights = Blender.tune_weights(prediction_models, train_df)
    print(best_weights)

    blender_model = Blender(models=prediction_models, weights=best_weights)
    blender_model.fit(train_df)
    pred = blender_model.predict(test_df)

    print("============\n[LOG] SAVE RESULT IN CSV FILE\n============")
    create_submission_file(pred, str("output.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--sample', type=int, default=0, help='Help text!')
    args = parser.parse_args()
    main(args)
