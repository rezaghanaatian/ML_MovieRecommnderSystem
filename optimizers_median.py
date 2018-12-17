from optimizers import optimizer


class GlobalMedian(optimizer):

    @staticmethod
    def predict(train, test):
        """
    Define the global median model for recommendation.

    Args:
        train (Pandas Dataframe) : train dataset
        test (Pandas Dataframe): test dataset

    Returns:
        output (Pandas Dataframe): test dataset with updated predictions calculated with global mean
    """

        output = test.copy()  # calculate the global median
        global_median_train = train.Prediction.median()

        output.Prediction = global_median_train

        def round_pred(row):
            return round(row.Prediction)

        output['Prediction'] = output.apply(round_pred, axis=1)
        output['Rating'] = output['Prediction']

        return output
