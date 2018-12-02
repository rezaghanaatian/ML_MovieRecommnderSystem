from models import global_mean, user_mean, movie_mean
from helpers import load_dataset, create_submission_file

""" Load dataset """

path_dataset = "data/data_train.csv"
path_test_dataset = "data/sample_submission.csv"

train_df = load_dataset(path_dataset)
test_df = load_dataset(path_test_dataset)

""" Use different models for prediction """
pred_global_mean = global_mean(train_df, test_df)
pred_user_mean = user_mean(train_df, test_df)
pred_movie_mean = movie_mean(train_df, test_df)

""" Combine different models to better predict """

""" Create submission file """
sub_df = create_submission_file(pred_global_mean)
