from abc import abstractmethod


class PredictionModel:
    """
    Abstract class for predictions
    """

    def __init__(self, train_df):
        """
        Args:
            train (Pandas Dataframe) : train dataset with headers: ['User','Movie','Prediction']
        """
        self.train_df = train_df.copy()

    @classmethod
    def get_name(cls):
        return cls.__name__

    @abstractmethod
    def predict(self, test):
        """
        Define the global median model for recommendation.

        Args:
            test (Pandas Dataframe): test dataset with headers: ['User','Movie','Prediction']

        Returns:
            output (Pandas Dataframe): test dataset with updated predictions with headers: ['User','Movie','Prediction']
        """
        pass

    @abstractmethod
    def cross_validate(self, k_fold=5):
        """
        Does cross validation on train_set

        Args:
            k_folds (Integer) : number of folds used in cross validation

        Returns:
            output (float): Average RMSE in cross-validation
        """
        pass

    @staticmethod
    def round_prediction(row):
        return round(row.Prediction)
