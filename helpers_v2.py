import numpy as np
import pandas as pd



def load_dataset(path_dataset):
    """
        Clean initial dataset, split user id and movie id to put them in new columns.

        Input:
            path_dataset (String): path to the csv file containing initial data

        Output:
            (Pandas Dataframe): Dataframe containing this columns: [index, Prediction, User, Movie]

    """

    df = pd.read_csv(path_dataset, sep=',')

    # Split id to user and movie
    user_movie_df = pd.DataFrame(df.Id.str.split('_', 1).tolist(), columns=['User', 'Movie'])

    # Map each user-movie to their corresponding prediction
    output = df.join(user_movie_df)
    output = output.drop(columns=['Id'])

    # Remove "r" before id of each user
    output['User'] = output['User'].map(lambda x: x.lstrip('r').rstrip('')).astype(int)

    # Remove "c" before id of each user
    output['Movie'] = output['Movie'].map(lambda x: x.lstrip('c').rstrip('')).astype(int)

    return output


def split_data(X, folds):
    """
        This function will return two array including n (number of folds) different array which contains datasets
        with almost same size in order to be used in cross-validation

        Input:
            X (Pandas Dataframe): input data
            folds (Integer): number of folds for cross validation or partitioning of the data

        Output:
            (Array) An array including n arrays
    """

    X = X.sample(frac=1).reset_index(drop=True)

    index_split_start = 0
    index_split_end = int(np.floor(1 / folds * len(X)))

    X_output = []

    for i in range(folds):
        if i == folds - 1:
            index_split_end = len(X)

        X_subset = X[index_split_start:index_split_end]
        X_output.append(X_subset)
        index_split_start = index_split_end
        index_split_end += len(X_subset)

    return X_output


def compute_error(data_gt, data_pred):
    """
        compute the loss (MSE) of the prediction.

        Input:
            data_gt: (Pandas Dataframe) Data ground truth
            data_pred: (Pandas Dataframe) Data prediction
    """
    if ~(data_gt.columns==['Prediction', 'User', 'Movie']).all():
        data_gt = dataFrameRecover(data_gt)
    
    mse = 0
    for index, row in data_gt.iterrows():
        mse += (data_gt.Prediction[index] - data_pred.Prediction[index]) ** 2
    return np.sqrt(1.0 * mse / len(data_gt))


def calculate_error(truth, prediction):
    """ compute RMSE for pandas.DataFrame prediction """
    
    truth_sorted = truth.sort_values(['Movie', 'User']).reset_index(drop=True)
    prediction_sorted = prediction.sort_values(['Movie', 'User']).reset_index(drop=True)

    truth_sorted['square_error'] = np.square(truth_sorted['Prediction'] - prediction_sorted['Prediction'])

    mse = truth_sorted['square_error'].mean()
    rmse = np.sqrt(mse)

    return rmse


def cross_validator(model, dataset, n_fold):
    """
        Split dataset to some folds and do cross validation with input model

        Input:
            model: (Function) Data ground truth
            dataset: (Pandas Dataframe) Data prediction
            n_folds: (Integer) number of folds

        Output:
            (Float) cross-validated error of prediction using input model
    """

    X_s = split_data(dataset, n_fold)
    errors = []
    for i in range(n_fold):
        print('========== Fold {} of {} =========='.format(i+1, n_fold))
        X_test = X_s[i]
        X_train = pd.DataFrame(columns=X_test.columns)

        for j in range(n_fold):
            if i == j:
                continue
            X_train = X_train.append(X_s[j], ignore_index=True)
            X_train = X_train.astype('int64')
        
        pred = model.predict(X_train, X_test)
        err = calculate_error(X_test, pred)
        print('error = {}\n'.format(err))
        errors.append(err)

    return np.mean(errors)


def blender(models, weights):
    """
    Blend different models using different weights

    Args:
        models (dict): keys: model name; values: pandas.DataFrame predicitons
        weights (dict): keys: model name; values: weights

    Returns:
        belnd (pandas.DataFrame): datafeame containing weighted predictions blended
    """
    if len(models) != len(weights):
        print("[WARNING] size(predictions) != size(weights)")

    # initiate a DF with desired user/movie key and null prediction
    blend = unified_ordering(list(models.values())[0])
    blend['Rating'] = 0.

    # sum weighted predictions
    for model in models.keys():
        blend['Rating'] += \
            weights[model] * unified_ordering(models[model])['Rating']
            
    pred = list(blend['Rating'])
    
    for i in range(len(pred)):
        if pred[i] > 5:
            pred[i] = 5
        elif pred[i] < 1:
            pred[i] = 1
    
    blend['Rating'] = pred

    return blend


def create_submission_file(prediction, file_name, output_folder):
    """
        Generate submission file for uploading on Kaggle and save it in output folder.

        Input:
            prediction (Pandas Dataframe): predicted values for test dataset
            output_name (String): Name of output file

        Output:
            (Pandas Dataframe) : predicted values in standard format

    """
    
    output = prediction.copy()

    def get_id(row):
        return "r" + str(int(row.User)) + "_c" + str(int(row.Movie))
    
    pred = list(output['Prediction'])
    
    for i in range(len(output)):
        if pred[i] > 5:
            pred[i] = 5
        elif pred[i] < 1:
            pred[i] = 1
            
    output['Prediction'] = pred
    output['Id'] = output.apply(get_id, axis=1)
    output[['Id', 'Prediction']].to_csv(output_folder + file_name, index=False)
    
    
    return output[['Id', 'Prediction']]

def time_str(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    
    if h > 0:
        str_ = "%d hour %d min. %d sec."%(h, m, s)
    elif m > 0:
        str_ = "%d min. %d sec."%(m, s)
    else:
        str_ = "%.3f sec."%s
    return str_ 


def dataFrameConvert(df):
    """
    This method Prepares the dataFrame to be used in certain parts of the code such as 
    normalizer or ALS object instantiation with headings ['index','User','Movie','Rating']
    
    Args:
       df: pandas dataframe with columns: ['index', 'Prediction', 'User', 'Movie']
       
        
    Returns:
        df: pandas dataframe with columns: ['index','User','Movie','Rating']
        
    """
    df = df.rename(columns={"Prediction": "Rating"})
    df = df[['User','Movie','Rating']]
    
    return df


def dataFrameRecover(df):
    """
    This method Prepares the dataFrame to be used in certain parts of the code such as parts other than
    normalizer or ALS object instantiation with headings ['index', 'Prediction', 'User', 'Movie']
    
    Args:
       df: pandas dataframe with columns: ['index','User','Movie','Rating']
       
    Returns:
        df : pandas dataframe with columns: ['index', 'Prediction', 'User', 'Movie']
        
    """
    df = df.rename(columns={"Rating": "Prediction"})
    df = df[['Prediction','User','Movie']]
   
    return df

