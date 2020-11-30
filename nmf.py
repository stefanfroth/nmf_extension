# %%
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
    return np.random.random_sample(size=(nr_of_users, n_components))

def initialize_item_feature_matrix(n_components, nr_of_items):
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
    return np.random.random_sample(size=(n_components, nr_of_items))


def update_user_feature_matrix(user_feature_matrix, item_feature_matrix, alpha, errors):
    '''
    Calculate the updates for user_feature_matrix.

    Parameters
    ----------
    user_feature_matrix : numpy.ndarray
        The user_feature_matrix to update.
    item_feature_matrix : numpy.ndarray
        The item_feature_matrix for update calculation.
    alpha : float
        The learning rate.
    errors : np.ndarray
        The errors of the last iteration.
    
    Returns
    -------
    np.ndarray
        The updated user_feature_matrix.
    '''
    # TODO: why does np.matmul not work in this case?
    updates = 2*alpha*np.dot(errors, item_feature_matrix.transpose())
    # updates[np.isnan(updates)] = 0
    user_feature_matrix -= updates
    return user_feature_matrix # user_feature_matrix.values


def update_item_feature_matrix(item_feature_matrix, user_feature_matrix, alpha, errors):
    '''
    Calculate the updates for user_feature_matrix.

    Parameters
    ----------
    item_feature_matrix : numpy.ndarray
        The item_feature_matrix to update.
    user_feature_matrix : numpy.ndarray
        The user_feature_matrix for update calculation.
    alpha : float
        The learning rate.
    errors : np.ndarray
        The errors of the last iteration.
    
    Returns
    -------
    np.ndarray
        The updated user_feature_matrix.
    '''
    # TODO: why does np.matmul not work in this case?
    updates = 2*alpha*np.dot(user_feature_matrix.transpose(), errors)
    # updates[np.isnan(updates)] = 0
    item_feature_matrix -= updates
    return item_feature_matrix


class CustomNMF(NMF):

    def fit_transform(self, X, y=None, W=None, H=None):
        if X.isna().any().any():
            W = initialize_user_feature_matrix(X.shape[0], self.n_components)
            H = initialize_item_feature_matrix(self.n_components, X.shape[1])
            Rhat = np.dot(W, H)
            errors = Rhat - X
            errors[np.isnan(errors)] = 0
            for i in range(self.max_iter):
                W = update_user_feature_matrix(W, H, self.alpha, errors)
                H = update_item_feature_matrix(H, W, self.alpha, errors)
                Rhat = np.matmul(W, H)
                errors = Rhat - X
                errors[np.isnan(errors)] = 0
                # print(f'We are in iteration {i}')
                # print(f'The mse is {round(np.nansum(errors**2), 2)}')

            self.n_components_ = H.shape[0]
            self.components_ = H
            self.n_iter_ = self.max_iter
            return np.round(Rhat, 2)
        super().fit_transform(X, y=y, W=W, H=H)