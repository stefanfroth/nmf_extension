'''Run a basic test to check if the test data has missing values'''

import pytest # later for pytest.raise
import numpy as np

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
