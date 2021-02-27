'''Run a basic test to check if the test data has missing values'''
import logging

import pytest # later for pytest.raise
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import NMF

LOGGER = logging.getLogger(__name__)

# ## TODO: Implement the number of Components as a fixture of its own?
def fit_model(model, R):
    '''Fit a model with R'''
    P_hat = model.fit_transform(R)
    R_hat = np.matmul(P_hat, model.components_)
    return R_hat


def test_missing(test_data):
    '''Test whether the test data actually has missing values'''
    R_missing = np.copy(test_data[1])
    assert np.isnan(R_missing).sum() > 0

def test_fit_transform(p_hat):
    '''
    Tests whether the CustomNMF class can be instantiated and the
    .fit_transform() method works.
    '''
    assert isinstance(p_hat, np.ndarray)

def test_transform(test_model, n_items):
    '''Tests whether an input array with null values can be transformed'''
    # R_missing = np.copy(test_data[1])
    # test_model.transform(R_missing)
    new_user = np.ones((1,n_items))
    mask = np.random.randint(n_items)
    new_user[:,mask] = np.nan
    assert isinstance(np.matmul(test_model.transform(new_user), test_model.components_), np.ndarray)


def test_shape(r_hat, test_data):
    '''Tests whether the resulting R_hat will have the right shape'''
    R_missing = np.copy(test_data[1])
    # P_hat = test_model.fit_transform(R_missing)
    # R_hat = np.matmul(p_hat, test_model.components_)
    assert r_hat.shape == R_missing.shape

## TODO: Test whether the mse remains unchanged for the movie dataset as well
## TODO: For now it tests that the CustomNMF is not performing worse
@pytest.mark.parametrize('impute', list(range(6)))
def test_better(test_data, mse_nmf, r_hat, impute, components, iterations):
    '''
    Test whether the CustomNMF actually performs better than an imputation
    with 0s
    '''
    # Extract test data
    R = np.copy(test_data[0])
    R_missing = np.copy(test_data[1])

    # Calculate R_hat with CustomNMF
    # R_hat = np.matmul(p_hat, test_model.components_)
    # mse_custom = mean_squared_error(R[~np.isnan(R)], r_hat[~np.isnan(R)])
    #mse_custom = mean_squared_error(R, R_hat)
    logging.info(f'The mean squared error of this technique is {mse_nmf}')

    # Calculate R_hat with zero imputation
    R_missing[np.isnan(R_missing)] = impute
    ## TODO: make the number of components flexible
    # classic_nmf = NMF(n_components=4)
    # P_hat_zero_imputed = classic_nmf.fit_transform(R_missing)
    # R_hat_imputed = np.matmul(P_hat_zero_imputed, classic_nmf.components_)
    R_hat_imputed = fit_model(NMF(n_components=components, max_iter=iterations), R_missing)
    mse_imputed = mean_squared_error(R[~np.isnan(R)], R_hat_imputed[~np.isnan(R)])
    #mse_imputed = mean_squared_error(R, R_hat_imputed)
    logging.info(f'The mean squared error of the {impute} imputed variant is {mse_imputed}')

    # Assert that the algorithm works better
    assert ~(round(mse_nmf, 2) > round(mse_imputed, 2))

