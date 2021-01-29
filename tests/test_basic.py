'''Run a basic test to check if the test data has missing values'''
import logging

import pytest # later for pytest.raise
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import NMF

LOGGER = logging.getLogger(__name__)

## TODO: Implement the number of Components as a fixture of its own?
def fit_model(model, R):
    '''Fit a model with R'''
    P_hat = model.fit_transform(R)
    R_hat = np.matmul(P_hat, model.components_)
    return R_hat


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

## TODO: Test whether the mse remains unchanged for the movie dataset as well
@pytest.mark.parametrize('impute', list(range(6)))
def test_better(test_data, test_model, impute, components, iterations):
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
    R_missing[np.isnan(R_missing)] = impute
    ## TODO: make the number of components flexible
    # classic_nmf = NMF(n_components=4)
    # P_hat_zero_imputed = classic_nmf.fit_transform(R_missing)
    # R_hat_imputed = np.matmul(P_hat_zero_imputed, classic_nmf.components_)
    R_hat_imputed = fit_model(NMF(n_components=components, max_iter=iterations), R_missing)
    mse_imputed = mean_squared_error(R, R_hat_imputed)
    logging.info(f'The mean squared error of the {impute} imputed variant is {mse_imputed}')

    # Assert that the algorithm works better
    assert mse_custom <= mse_imputed

