import copy
import numpy as np
import pandas as pd
from abc import abstractmethod
from helpers import split_data, compute_error


class PredictionModel:
    """
    Abstract class for predictions
    """

    def __init__(self):
        """
        Args:
            train (Pandas Dataframe) : train dataset with headers: ['User','Movie','Prediction']
        """
        self.train_df = None
        self.model = None

    @classmethod
    def get_name(cls):
        return cls.__name__

    def fit(self, train_df):
        """
        store training data for the model. this function can be overridden in the case
        a manipulation is needed in training data.

        Args:
            test (Pandas Dataframe): test dataset with headers: ['User','Movie','Prediction']

        Returns:
            output (Pandas Dataframe): test dataset with updated predictions with headers: ['User','Movie','Prediction']
        """
        self.train_df = train_df.copy()

    @abstractmethod
    def predict(self, test_df):
        """
        Predict rating for test data based on trained model

        Args:
            test (Pandas Dataframe): test dataset with headers: ['User','Movie','Prediction']

        Returns:
            output (Pandas Dataframe): test dataset with updated predictions with headers: ['User','Movie','Prediction']
        """
        pass

    def cross_validate(self, k_fold=5):
        """
        Does cross validation on train_set

        Args:
            k_folds (Integer) : number of folds used in cross validation

        Returns:
            output (float): Average RMSE in cross-validation
        """

        if self.train_df is None:
            raise Exception('You should first fit the model!')

        X_s = split_data(self.train_df, k_fold)
        errors = []
        for i in range(k_fold):
            X_test = X_s[i]
            X_train = pd.DataFrame(columns=X_test.columns)

            for j in range(k_fold):
                if i == j:
                    continue
                X_train = X_train.append(X_s[j], ignore_index=True)

            model = copy.deepcopy(self)
            model.fit(X_train)
            pred = model.predict(X_test)
            err = compute_error(X_test, pred)
            errors.append(err)

        return np.mean(errors)

    @staticmethod
    def round_prediction(row):
        return round(row.Prediction)
