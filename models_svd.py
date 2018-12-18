from surprise.prediction_algorithms import SVD, SVDpp
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
import scipy.sparse as sp
from helpers import df_to_sp, sp_to_df
from models_knn import SurpriseModel


class SurpriseSVD(SurpriseModel):
    n_epochs = 30
    n_factors = 10
    lr_all = 0.001
    reg_all = 0.01
   
    

    def __init__(self, n_epochs=n_epochs , n_factors=n_factors, lr_all=lr_all, reg_all=reg_all):
        super(SurpriseSVD, self).__init__()
         
        # set the parameters for SVD
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.lr_all = lr_all
        self.reg_all = reg_all

       
        self.model= SVD(n_factors=self.n_factors,n_epochs=self.n_epochs,lr_all=self.lr_all,reg_all=self.reg_all)
       
       
    def cross_validate(self, k_fold=5):
        return super(SurpriseSVD, self).cross_validate(k_fold)

    def predict(self, test):
        return super(SurpriseSVD, self).predict(test)


from surprise.prediction_algorithms import SVD
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
import scipy.sparse as sp
from models_knn import SurpriseModel
from helpers import df_to_sp, sp_to_df


class SurpriseSVDpp(SurpriseModel):
    n_epochs = 30
    n_factors = 10
    lr_all = 0.001
    reg_all = 0.01
   
    

    def __init__(self, n_epochs=n_epochs , n_factors=n_factors, lr_all=lr_all, reg_all=reg_all):
        super(SurpriseSVDpp, self).__init__()
         
        # set the parameters for SVD
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.lr_all = lr_all
        self.reg_all = reg_all

       
        self.model= SVDpp(n_factors=self.n_factors,n_epochs=self.n_epochs,lr_all=self.lr_all,reg_all=self.reg_all)
      

       
    def cross_validate(self, k_fold=5):
        return super(SurpriseSVDpp, self).cross_validate(k_fold)

    def predict(self, test):
        return super(SurpriseSVDpp, self).predict(test)


