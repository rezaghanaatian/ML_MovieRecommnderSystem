import os
import numpy as np
from prediction_model import PredictionModel
from helpers import df_to_sp, sp_to_df

class SGDOptimizer(PredictionModel):
    """
        A class for SGD optimizer implementation 
    """
    gamma = 0.02
    nb_epochs = 20
    nb_latent = 30
    lambda_user = 0.1
    lambda_movie = 0.7
    
    def __init__(self, gamma=gamma, nb_epochs=nb_epochs, nb_latent=nb_latent, lambda_user=lambda_user,lambda_movie =lambda_movie):
        
        super(SGDOptimizer, self).__init__()
        self.model = None
        self.gamma = gamma
        self.nb_epochs = nb_epochs
        self.nb_latent = nb_latent
        self.lambda_user = lambda_user
        self.lambda_movie = lambda_movie



    def predict(self, test):
        """
        Define the global als model for recommendation.

        Args:
            train (Pandas Dataframe) : train dataset
            test (Pandas Dataframe): test dataset

        Returns:
            output (Pandas Dataframe): test dataset with updated predictions calculated with global mean
        """
        output = test.copy()
        D = max(np.max(self.train_df.Movie),np.max(output.Movie))
        N = max(np.max(self.train_df.User),np.max(output.User))
        K = self.nb_latent 
        np.random.seed(988)
        movie_features = np.random.rand(D,K)
        user_features = np.random.rand(N,K)

        # convert to scipy.sparse matrices
        train_sp = df_to_sp(self.train_df)

        # find the non-zero indices
        nz_row, nz_col = train_sp.nonzero()
        nz_train = list(zip(nz_row, nz_col))

        # the gradient loop
        for it in range(self.nb_epochs):
            # shuffle the training rating indices
            np.random.shuffle(nz_train)

            # decrease step size
            self.gamma /= 1.2

            for d, n in nz_train:
                # matrix factorization.
                err = train_sp[d, n] - np.dot(movie_features[d, :], user_features[n, :].T)
                grad_movie = -err * user_features[n, :] + self.lambda_movie * movie_features[d, :]
                grad_user = -err * movie_features[d, :] + self.lambda_user * user_features[n, :]

                movie_features[d, :] -= self.gamma * grad_movie
                user_features[n, :] -= self.gamma * grad_user

        # do the prediction and fill the test set
        test_sp = df_to_sp(output)
        nz_row, nz_col = test_sp.nonzero()
        nz_test = list(zip(nz_row, nz_col))
        X_pred = np.round(np.dot(movie_features, user_features.T))
        for row, col in nz_test:
            val = X_pred[row, col]
#            if val > 5:
#                pred= 5
#            elif val < 1:
#                pred = 1
#            else:
#                pred = val
            test_sp[row, col] = pred

        test_pred = sp_to_df(test_sp)
        print(test_pred)
        return test_pred