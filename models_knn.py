import numpy as np
from surprise import Reader, Dataset, KNNWithMeans, KNNBaseline
from surprise.model_selection import cross_validate, GridSearchCV

from prediction_model import PredictionModel


class SurpriseModel(PredictionModel):
    """
        A class for using surprise library
    """

    def __init__(self):
        super(SurpriseModel, self).__init__()
        self.model = None

    def fit(self, train_df):
        reader = Reader(rating_scale=(1, 5))
        train_data = Dataset.load_from_df(train_df[['User', 'Movie', 'Prediction']], reader)
        self.train_df = train_data

    def predict(self, test):
        """
        Define the global median model for recommendation.

        Args:
            train (Pandas Dataframe) : train dataset
            test (Pandas Dataframe): test dataset

        Returns:
            output (Pandas Dataframe): test dataset with updated predictions calculated with global mean
        """
        if self.model:
            self.model.fit(self.train_df.build_full_trainset())
        else:
            print("model has not been initialized.")

        output = test.copy()
        for index, row in output.iterrows():
            prediction = self.model.predict(row.User, row.Movie).est
            if prediction > 5:
                prediction = 5
            if prediction < 1:
                prediction = 1
            row.Prediction = round(prediction)
        return output

    def cross_validate(self, k_fold=5):
        return np.mean(cross_validate(self.model, self.train_df, measures=['RMSE', 'MAE'], cv=k_fold)['test_rmse'])


class SurpriseKNN(SurpriseModel):
    neighbors_num = 40
    is_user_based = True
    use_baseline = True

    def __init__(self, k=neighbors_num, user_based=is_user_based, baseline=use_baseline):
        super(SurpriseKNN, self).__init__()
        self.neighbors_num = k
        self.is_user_based = user_based
        self.use_baseline = baseline
        if self.use_baseline:
            self.model = KNNBaseline(k=self.neighbors_num,
                                     sim_options={'name': 'pearson_baseline', 'user_based': self.is_user_based})
        else:
            self.model = KNNWithMeans(k=self.neighbors_num,
                                      sim_options={'user_based': self.is_user_based})

    def cross_validate(self, k_fold=5):
        return super(SurpriseKNN, self).cross_validate(k_fold)

    def predict(self, test):
        return super(SurpriseKNN, self).predict(test)

    @staticmethod
    def tune_weights(train_df):
        """
        Print best hyper parameters for the model
        :param train_df: train data for calculating RMSE by cross-validation
        :return: the found best weights for given models
        """

        param_grid = {'bsl_options': {'method': ['als', 'sgd']},
                      'k': [40, 60, 90, 100, 200, 300],
                      'sim_options': {'name': ['msd', 'cosine', 'pearson_baseline'],
                                      'user_based': [False, True]}}

        gs = GridSearchCV(KNNBaseline, param_grid, measures=['rmse', 'MAE'], cv=5)
        reader = Reader(rating_scale=(1, 5))
        train_data = Dataset.load_from_df(train_df[['User', 'Movie', 'Prediction']], reader)
        gs.fit(train_data)

        print("Best rmse:".format(gs.best_score['rmse']))

        # combination of parameters that gave the best RMSE score
        print("Best hyper parameters:".format(gs.best_params['rmse']))
