import numpy as np
import pandas as pd
import scipy.sparse as sp


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
    #output = output.drop(columns=['Id'])
    output = output.drop(labels=["Id"], axis=1)
    
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
        compute the loss (RMSE) of the prediction.

        Input:
            data_gt: (Pandas Dataframe) Data ground truth
            data_pred: (Pandas Dataframe) Data prediction
    """
    mse = 0
    for index, row in data_gt.iterrows():
        mse += (data_gt.Prediction[index] - data_pred.Prediction[index]) ** 2
    return np.sqrt(1.0 * mse / len(data_gt))


def compute_error2(data_gt, data_pred):
    """
        compute the loss (RMSE) of the prediction.

        Input:
            data_gt: (Pandas Dataframe) Data ground truth
            data_pred: (Pandas Dataframe) Data prediction
    """
    mse = 0
    gt_sp = df_to_sp(data_gt)
    nz_row, nz_col = gt_sp.nonzero()
    nz_gt = list(zip(nz_row, nz_col))
    pred_sp = df_to_sp(data_pred)
    for d, n in nz_gt:
         mse += (gt_sp[d,n] - pred_sp[d,n]) **2
    return np.sqrt(1.0 * mse / len(data_gt))


def cross_validator(model, dataset, n_fold=5):
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
        X_test = X_s[i]
        X_train = pd.DataFrame(columns=X_test.columns)

        for j in range(n_fold):
            if i == j:
                continue
            X_train = X_train.append(X_s[j], ignore_index=True)

        pred = model(X_train, X_test)
        err = compute_error2(X_test, pred)
        errors.append(err)

    return np.mean(errors)


def cross_validator_param_opt(model, dataset, n_fold, **kwargs):
    """
        Split dataset to some folds and do cross validation with input model

        Input:
            model: (Function) Data ground truth
            dataset: (Pandas Dataframe) Data prediction
            n_folds: (Integer) number of folds
            **kwarg: arguments to be passed to the model

        Output:
            (Float) cross-validated error of prediction using input model
    """

    X_s = split_data(dataset, n_fold)
    errors = []
    for i in range(n_fold):
        X_test = X_s[i]
        X_train = pd.DataFrame(columns=X_test.columns)

        for j in range(n_fold):
            if i == j:
                continue
            X_train = X_train.append(X_s[j], ignore_index=True)

        pred = model(X_train, X_test, **kwargs)
        err = compute_error2(X_test, pred)
        errors.append(err)

    return np.mean(errors)


def create_submission_file(prediction, output_name="submission.csv"):
    """
        Generate submission file for uploading on Kaggle and save it in output folder.

        Input:
            prediction (Pandas Dataframe): predicted values for test dataset
            output_name (String): Name of output file

        Output:
            (Pandas Dataframe) : predicted values in standard format

    """
    output_folder = "output/"
    output = prediction.copy()

    def get_id(row):
        return "r" + str(int(row.User)) + "_c" + str(int(row.Movie))

    def round_pred(row):
        return round(row.Prediction)

    output['Id'] = output.apply(get_id, axis=1)
    output['Prediction'] = output.apply(round_pred, axis=1)

    output[['Id', 'Prediction']].to_csv(output_folder + output_name, index=False)
    return output[['Id', 'Prediction']]


def df_to_sp(df):
    """ Convert a pandas.DataFrame to a scipy.sparse matrix """

    n_user = df['User'].max()
    n_movie = df['Movie'].max()

    sp_matrix = sp.lil_matrix((n_movie, n_user))

    users = df['User']
    movies = df['Movie']
    prediction = df['Prediction']

    for m, u, p in zip(movies, users, prediction):
        sp_matrix[m - 1, u - 1] = p

    return sp_matrix

def sp_to_df(sparse):
    """ Convert scipy.sparse matrix to pandas.DataFrame """

    row, col, pred = sp.find(sparse)
    row += 1
    col += 1

    df = pd.DataFrame({'Prediction': pred, 'User': col, 'Movie': row})
    df = df[['Prediction', 'User', 'Movie']].sort_values(['Movie', 'User'])
    df=df.reset_index(drop=True)
    return df
