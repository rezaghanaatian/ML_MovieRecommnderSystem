class Normalizer(object):
    """
    Normalize the training set and predictions
    """

    USER = 'user'
    MOVIE = 'movie'
    COMBINED = 'combined'
    GLOBAL = 'global'

    valid_types = [USER, MOVIE, COMBINED, GLOBAL]

    def __init__(self, normalization_type=COMBINED):
        """
        """
        if normalization_type not in Normalizer.valid_types:
            raise Exception('The type is not valid!')

        self.normalization_type = normalization_type
        self.mean_prediction_by_user = None
        self.mean_prediction_by_movie = None
        self.global_mean = None

    def normalize(self, train_df):

        self.mean_prediction_by_user = train_df.groupby('User')[['Prediction']].mean()
        self.mean_prediction_by_movie = train_df.groupby('Movie')[['Prediction']].mean()
        self.global_mean = train_df.Prediction.mean()

        output = train_df.copy()

        def assign_mean(row):
            movie = row.Movie
            user = row.User
            mean_user = self.mean_prediction_by_user[self.mean_prediction_by_user.index == user].iloc[0].Prediction
            mean_movie = self.mean_prediction_by_movie[self.mean_prediction_by_movie.index == movie].iloc[0].Prediction

            pred = row.Prediction
            if self.normalization_type == Normalizer.USER:
                pred -= (self.global_mean - mean_user)
            elif self.normalization_type == Normalizer.MOVIE:
                pred -= (self.global_mean - mean_movie)
            elif self.normalization_type == Normalizer.GLOBAL:
                pred -= self.global_mean
            else:
                pred -= (self.global_mean - (mean_user + mean_movie))

            if pred > 5:
                return 5
            return pred if pred > 1 else 1

        output['Prediction'] = output.apply(assign_mean, axis=1)

        return output

    def revert_normalization(self, predictions):
        output = predictions.copy()

        def assign_mean(row):
            movie = row.Movie
            user = row.User
            mean_user = self.mean_prediction_by_user[self.mean_prediction_by_user.index == user].iloc[0].Prediction
            mean_movie = self.mean_prediction_by_movie[self.mean_prediction_by_movie.index == movie].iloc[0].Prediction

            pred = row.Prediction
            if self.normalization_type == Normalizer.USER:
                pred += (self.global_mean - mean_user)
            elif self.normalization_type == Normalizer.MOVIE:
                pred += (self.global_mean - mean_movie)
            elif self.normalization_type == Normalizer.GLOBAL:
                pred += self.global_mean
            else:
                pred += (self.global_mean - (mean_user + mean_movie))

            if pred < 1:
                return 1

            return pred if pred < 5 else 5

        output['Prediction'] = output.apply(assign_mean, axis=1)

        return output
