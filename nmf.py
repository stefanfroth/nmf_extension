'''
Write my own algorithm that performs NMF on the matrix and returns the 
imputed matrix.
'''

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF

np.random.seed(10)

def initialize_user_feature_matrix(nr_of_users, n_components):
    '''
    Initializes the user_feature_matrix.

    Parameters
    ----------
    nr_of_users : int
        The numbers of users that are present in the 
    n_components : int
        The number of components the user_feature_matrix should have.
    
    Returns
    -------
    np.ndarray
        The user_feature_matrix
    '''
    return np.random.randint(1, 3, size=(nr_of_users, n_components)).astype(float)

def initialize_item_feature_matrix(nr_of_items, n_components):
    '''
    Initializes the user_feature_matrix.

    Parameters
    ----------
    nr_of_users : int
        The numbers of users that are present in the 
    n_components : int
        The number of components the user_feature_matrix should have.
    
    Returns
    -------
    np.ndarray
        The user_feature_matrix
    '''
    return np.random.randint(1, 3, size=(n_components, nr_of_items)).astype(float)


def update_user_feature_matrix(user_feature_matrix, item_feature_matrix, alpha, errors):
    # calculate the updates for user_feature_matrix
    user_feature_matrix = pd.DataFrame(user_feature_matrix)
    item_feature_matrix = pd.DataFrame(item_feature_matrix)
    errors = pd.DataFrame(errors)
    for user in user_feature_matrix.index:
        for feature in user_feature_matrix.columns:
            update = 0
            old = user_feature_matrix.at[user, feature]
            for i, error in enumerate(errors.iloc[user]):
                if not pd.isna(error):
                    update -= 2*alpha*error*item_feature_matrix.loc[feature].iat[i]
            user_feature_matrix.at[user, feature] = old+update
    return user_feature_matrix.values


def update_item_feature_matrix(user_feature_matrix, item_feature_matrix, alpha, errors):
    # calculate the updates for user_feature_matrix
    user_feature_matrix = pd.DataFrame(user_feature_matrix)
    item_feature_matrix = pd.DataFrame(item_feature_matrix)
    errors = pd.DataFrame(errors)
    for feature in item_feature_matrix.index:
        for item in item_feature_matrix.columns:
            update = 0
            old = item_feature_matrix.at[feature, item]
            for i, error in enumerate(errors.iloc[:,item]):
                if not pd.isna(error):
                    update -= 2*alpha*error*user_feature_matrix.loc[:,feature].iat[i]
            item_feature_matrix.at[feature, item] = old+update
    return item_feature_matrix.values


class CustomNMF(NMF):

    def fit_transform(self, X, y=None, W=None, H=None):
        if X.isna().any().any():
            W = initialize_user_feature_matrix(X.shape[0], self.n_components)
            H = initialize_item_feature_matrix(X.shape[1], self.n_components)
            Rhat = np.dot(W, H)
            errors = Rhat - X
            for i in range(self.max_iter):
                W = update_user_feature_matrix(W, H, self.alpha, errors)
                H = update_item_feature_matrix(W, H, self.alpha, errors)
                Rhat = np.matmul(W, H)
                errors = Rhat - X
                print(f'We are in iteration {i}')
                print(f'The mse is {round(np.nansum(errors**2), 2)}')
            return np.round(Rhat, 2)
        super().fit_transform(X, y=y, W=W, H=H)