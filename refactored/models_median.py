from refactored.prediction_model import PredictionModel


class GlobalMedian(PredictionModel):
    """
        Define the global median model for recommendation.
    """

    def predict(self, test):
        output = test.copy()  # calculate the global median
        global_median_train = self.train_df.Prediction.median()

        output.Prediction = global_median_train
        return output

    def cross_validate(self, k_fold=5):
        pass


class UserMedian(PredictionModel):
    """
        Define the user median model for recommendation.
    """

    def predict(self, test):
        output = test.copy()
        self.train_df.User = self.train_df.User.astype(int)
        self.train_df.Prediction = self.train_df.Prediction.astype(int)

        # calculate the median for each user
        median_prediction_by_user = self.train_df.groupby('User')[['Prediction']].median()

        def assign_median(row):
            return median_prediction_by_user[median_prediction_by_user.index == row.User].iloc[0].Prediction

        output['Prediction'] = output.apply(assign_median, axis=1)
        return output

    def cross_validate(self, k_fold=5):
        pass


class MovieMedian(PredictionModel):
    """
        Define the movie median model for recommendation.
    """

    def predict(self, test):
        output = test.copy()
        self.train_df.Movie = self.train_df.Movie.astype(int)
        self.train_df.Prediction = self.train_df.Prediction.astype(int)

        # calculate the mean for each movie
        median_prediction_by_movie = self.train_df.groupby('Movie')[['Prediction']].median()

        def assign_median(row):
            return median_prediction_by_movie[median_prediction_by_movie.index == row.Movie].iloc[0].Prediction

        output['Prediction'] = output.apply(assign_median, axis=1)
        return output

    def cross_validate(self, k_fold=5):
        pass
