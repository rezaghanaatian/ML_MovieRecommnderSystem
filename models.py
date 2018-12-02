def global_mean(train, test):
    """
        Define the global mean model for recommendation.

        Input:
            train (Pandas Dataframe) : train dataset
            test (Pandas Dataframe): test dataset

        Output:
            (Pandas Dataframe): test dataset with updated predictions calculated with global mean

    """

    output = test.copy()

    # calculate the global mean
    global_mean_train = train.Prediction.mean()

    output.Prediction = global_mean_train

    def round_pred(row):
        return round(row.Prediction)

    output['Prediction'] = output.apply(round_pred, axis=1)
    return output


def user_mean(train, test):
    """
        Define the user mean model for recommendation. recommends based on user's average rating.

        Input:
            train (Pandas Dataframe) : train dataset
            test (Pandas Dataframe): test dataset

        Output:
            (Pandas Dataframe): test dataset with updated predictions calculated with global mean

    """

    output = test.copy()
    train.User = train.User.astype(int)
    train.Prediction = train.Prediction.astype(int)

    # calculate the mean for each user
    mean_pred_by_user = train.groupby('User')[['Prediction']].mean()

    def assign_mean(row):
        return mean_pred_by_user[mean_pred_by_user.index == row.User].iloc[0].Prediction

    def round_pred(row):
        return round(row.Prediction)

    output['Prediction'] = output.apply(assign_mean, axis=1)
    output['Prediction'] = output.apply(round_pred, axis=1)
    return output


def movie_mean(train, test):
    """
        Define the user mean model for recommendation. recommends based on movie's average rating.

        Input:
            train (Pandas Dataframe) : train dataset
            test (Pandas Dataframe): test dataset

        Output:
            (Pandas Dataframe): test dataset with updated predictions calculated with movie mean

    """

    output = test.copy()
    train.Movie = train.Movie.astype(int)
    train.Prediction = train.Prediction.astype(int)

    # calculate the mean for each movie
    mean_pred_by_movie = train.groupby('Movie')[['Prediction']].mean()

    def assign_mean(row):
        return mean_pred_by_movie[mean_pred_by_movie.index == row.Movie].iloc[0].Prediction

    def round_pred(row):
        return round(row.Prediction)

    output['Prediction'] = output.apply(assign_mean, axis=1)
    output['Prediction'] = output.apply(round_pred, axis=1)
    return output
