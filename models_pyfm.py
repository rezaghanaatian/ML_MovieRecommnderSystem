import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction import DictVectorizer
from pyfm import pylibfm
from prediction_model import PredictionModel



class PyFmModel(PredictionModel):
    """
        A class for using PyFm library
    """

    def __init__(self):
        super(PyFmModel, self).__init__()
        self.model = None

    def fit(self, train_df):
        
        def prepare_data(df):
            """
            Prepare the data for the specific format used by PyFM.

            Args:
                df (pd.DataFrame): Initial DataFrame to transform

            Returns:
                data (array[dict]): Array of dict with user and movie ids
                y (np.array): Ratings give in the initial pd.DataFrame
                users (set): Set of user ids
                movies (set): Set of movie ids

            """
            data = []
            y = list(df.Prediction)
            users = set(df.User.unique())
            movies = set(df.Movie.unique())
            usrs = list(df.User)
            mvies = list(df.Movie)
            for i in range(len(df)):
                y[i] = float(y[i])
                data.append({"user_id": str(usrs[i]), "movie_id": str(mvies[i])})
            return (data, np.array(y), users, movies)
        
        (train_data, train_pred, train_users, train_movies) = prepare_data(train_df)
        v = DictVectorizer()
        x_train = v.fit_transform(train_data)
        
        self.x_train= x_train
        self.train_pred = train_pred
        self.train_df = train_df

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
            
            self.model.fit(self.x_train,self.train_pred)
    
        else:
            print("model has not been initialized.")



class PyFmOptimizer(PyFmModel):
    
    n_epochs = 200
    n_factors = 20
    lr_all = 0.002
    task = 'regression'
    learning_schedule = 'optimal'
    
    def __init__(self, n_epochs=n_epochs, n_factors=n_factors, lr_all=lr_all, task=task, learning_schedule=learning_schedule):
        super(PyFmOptimizer, self).__init__()
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.lr_all = lr_all
        self.task = task
        self.learning_schedule = learning_schedule

        
         # Create the FM model
        self.model= pylibfm.FM(num_factors=self.n_factors, num_iter=self.n_epochs, task=self.task, initial_learning_rate=self.lr_all, learning_rate_schedule=self.learning_schedule, verbose=False)
        
    def predict(self, test):
         
        output = test.copy()
        
        def prepare_data(df):
            """
            Prepare the data for the specific format used by PyFM.

            Args:
                df (pd.DataFrame): Initial DataFrame to transform

            Returns:
                data (array[dict]): Array of dict with user and movie ids
                y (np.array): Ratings give in the initial pd.DataFrame
                users (set): Set of user ids
                movies (set): Set of movie ids

            """
            data = []
            y = list(df.Prediction)
            users = set(df.User.unique())
            movies = set(df.Movie.unique())
            usrs = list(df.User)
            mvies = list(df.Movie)
            for i in range(len(df)):
                y[i] = float(y[i])
                data.append({"user_id": str(usrs[i]), "movie_id": str(mvies[i])})
            
            return (data, np.array(y), users, movies)
        
        (test_data, test_pred, test_users, test_movies) = prepare_data(output)
      
        t = DictVectorizer()
        x_test = t.transform(test_data)
           
        # prediction
        preds = super(PyFmOptimizer, self).predict(x_test)
        vals = preds.copy()
    
        for i in range(len(preds)):
            if preds[i] > 5:
                vals[i] = 5
            elif preds[i] < 1:
                vals[i] = 1
            else:
                vals[i]=np.round(preds[i])
                
                
        test_pred = test.copy()
        test_pred.Prediction = vals    
        
       
        return test_pred