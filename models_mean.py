from prediction_model import PredictionModel


class GlobalMean(PredictionModel):
    """
        Define the global mean model for recommendation.
    """

    def predict(self, test):
        output = test.copy()
        # calculate the global mean
        global_mean_train = self.train_df.Prediction.mean()
        output.Prediction = global_mean_train
        return output


class UserMean(PredictionModel):
    """
        Define the user mean model for recommendation.
    """

    def predict(self, test):
        output = test.copy()
        self.train_df.User = self.train_df.User.astype(int)
        self.train_df.Prediction = self.train_df.Prediction.astype(int)

        # calculate the mean for each user
        mean_prediction_by_user = self.train_df.groupby('User')[['Prediction']].mean()

        def assign_mean(row):
            return mean_prediction_by_user[mean_prediction_by_user.index == row.User].iloc[0].Prediction

        output['Prediction'] = output.apply(assign_mean, axis=1)
        return output


class MovieMean(PredictionModel):
    """
        Define the movie mean model for recommendation.
    """

    def predict(self, test):
        output = test.copy()
        self.train_df.Movie = self.train_df.Movie.astype(int)
        self.train_df.Prediction = self.train_df.Prediction.astype(int)

        # calculate the mean for each movie
        mean_prediction_by_movie = self.train_df.groupby('Movie')[['Prediction']].mean()

        def assign_mean(row):
            return mean_prediction_by_movie[mean_prediction_by_movie.index == row.Movie].iloc[0].Prediction

        output['Prediction'] = output.apply(assign_mean, axis=1)
        return output
