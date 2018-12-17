from refactored.prediction_model import PredictionModel


class GlobalMean(PredictionModel):
    """
        Define the global mean model for recommendation.
    """

    def predict(self, test):
        output = test.copy()
        # calculate the global mean
        global_mean_train = self.train_df.Prediction.mean()
        output.Prediction = round(global_mean_train)
        return output

    def cross_validate(self, k_fold=5):
        pass


class GlobalMean(PredictionModel):
    """
        Define the global mean model for recommendation.
    """

    def predict(self, test):
        output = test.copy()
        # calculate the global mean
        global_mean_train = self.train_df.Prediction.mean()
        output.Prediction = round(global_mean_train)
        return output

    def cross_validate(self, k_fold=5):
        pass
