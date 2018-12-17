from helpers import load_dataset, create_submission_file
from optimizers_knn import SurpriseKNN
from optimizers_median import GlobalMedian

""" Load dataset """

path_dataset = "data/data_train.csv"
path_test_dataset = "data/sample_submission.csv"

train_df = load_dataset(path_dataset)
test_df = load_dataset(path_test_dataset)

""" Use different models for prediction """
# pred_global_mean = global_mean(train_df, test_df)
# pred_user_mean = user_mean(train_df, test_df)
# pred_movie_mean = movie_mean(train_df, test_df)
# pred_movie_median = GlobalMedian.predict(train_df, test_df)

# Use KNN model (Scikit) to predict
# knn_model = UserKNN(200)
# pred_movie_mean = knn_model.predict(train_df, test_df)

model = SurpriseKNN(train_df, k=50, user_based=False)
pred_knn = model.cross_validate()
pred_knn.head()

""" Combine different models to better predict """

""" Create submission file """
# sub_df = create_submission_file(pred_movie_median, "submission_global_median.csv")
