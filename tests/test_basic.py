'''Run a basic test to check if the test data has missing values'''
import logging

import pytest # later for pytest.raise
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import NMF

LOGGER = logging.getLogger(__name__)

def test_missing(test_data):
    '''Test whether the test data actually has missing values'''
    R_missing = test_data[1]
    assert np.isnan(R_missing).sum() > 0

def test_running(test_data, test_model):
    '''Tests whether the CustomNMF function can be imported and run'''
    R_missing = test_data[1]
    assert isinstance(test_model.fit_transform(R_missing), np.ndarray)

def test_shape(test_data, test_model):
    '''Tests whether the resulting R_hat will have the right shape'''
    R_missing = test_data[1]
    P_hat = test_model.fit_transform(R_missing)
    R_hat = np.matmul(P_hat, test_model.components_)
    assert R_hat.shape == R_missing.shape

def test_better_zero_imputation(test_data, test_model):
    '''
    Test whether the CustomNMF actually performs better than an imputation
    with 0s
    '''
    # Extract test data
    R = test_data[0]
    R_missing = test_data[1]

    # Calculate R_hat with CustomNMF
    P_hat = test_model.fit_transform(R_missing)
    R_hat = np.matmul(P_hat, test_model.components_)
    mse_custom = mean_squared_error(R, R_hat)
    logging.info(f'The mean squared error of this technique is {mse_custom}')

    # Calculate R_hat with zero imputation
    R_missing[np.isnan(R_missing)] = 0
    ## TODO: make the number of components flexible
    classic_nmf = NMF(n_components=4)
    P_hat_zero_imputed = classic_nmf.fit_transform(R_missing)
    R_hat_zero_imputed = np.matmul(P_hat_zero_imputed, classic_nmf.components_)
    mse_zero_imputed = mean_squared_error(R, R_hat_zero_imputed)
    logging.info(f'The mean squared error of the zero imputed variant is {mse_zero_imputed}')

    # Assert that the algorithm works better
    assert mse_custom <= mse_zero_imputed

