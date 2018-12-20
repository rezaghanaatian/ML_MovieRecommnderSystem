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


class PersonalizedMovieMean(PredictionModel):
    """
        This model use movie mean as the basis of recommendation, but it also
        consider the behaviour of user in the rating. So it simply add user deviation
        to the movie mean for each prediction.
    """

    def predict(self, test):
        output = test.copy()
        self.train_df.Movie = self.train_df.Movie.astype(int)
        self.train_df.Prediction = self.train_df.Prediction.astype(int)

        mean_prediction_by_movie = self.train_df.groupby('Movie')[['Prediction']].mean()
        mean_prediction_by_user = self.train_df.groupby('User')[['Prediction']].mean()
        global_mean = self.train_df.Prediction.mean()

        def assign_mean(row):
            movie = row.Movie
            user = row.User

            mean_user = mean_prediction_by_user[mean_prediction_by_user.index == user].iloc[0].Prediction
            mean_movie = mean_prediction_by_movie[mean_prediction_by_movie.index == movie].iloc[0].Prediction

            return mean_movie - (global_mean - mean_user)

        output['Prediction'] = output.apply(assign_mean, axis=1)
        return output
