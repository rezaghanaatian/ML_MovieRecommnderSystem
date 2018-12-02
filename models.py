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
