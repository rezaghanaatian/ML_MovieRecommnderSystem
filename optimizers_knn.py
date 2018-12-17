from surprise import Reader, Dataset, KNNWithMeans, KNNBaseline
from surprise.model_selection import cross_validate


class SurpriseModel:
    """

    """

    def __init__(self, train_df):
        reader = Reader(rating_scale=(1, 5))
        train_data = Dataset.load_from_df(train_df[['User', 'Movie', 'Prediction']], reader)
        self.train_df = train_data
        self.model = None

    def predict(self, test):
        """
        Define the global median model for recommendation.

        Args:
            train (Pandas Dataframe) : train dataset
            test (Pandas Dataframe): test dataset

        Returns:
            output (Pandas Dataframe): test dataset with updated predictions calculated with global mean
        """
        if self.model:
            self.model.fit(self.train_df.build_full_trainset())
        else:
            print("model has not been initialized.")

        pred = test.copy()
        for index, row in pred.iterrows():
            pred.Prediction = round(self.model.predict(row.User, row.Movie, clip=True).est)

        return pred

    def cross_validate(self, k_fold=5):
        """
        Does cross validation on train_set

        Args:
            k_folds (Integer) : number of folds used in cross validation

        Returns:
            output (float): Average RMSE in cross-validation
        """
        cross_validate(self.model, self.train_df, measures=['RMSE', 'MAE'], cv=k_fold)





class SurpriseKNN(SurpriseModel):
    neighbors_num = 40
    is_user_based = True
    use_baseline = True

    def __init__(self, train, k=neighbors_num, user_based=is_user_based, baseline=use_baseline):
        super(SurpriseKNN, self).__init__(train_df=train)
        self.neighbors_num = k
        self.is_user_based = user_based
        self.use_baseline = baseline

        if self.use_baseline:
            self.model = KNNBaseline(k=self.neighbors_num,
                                     sim_options={'name': 'pearson_baseline', 'user_based': self.is_user_based})
        else:
            self.model = KNNWithMeans(k=self.neighbors_num,
                                      sim_options={'user_based': self.is_user_based})
