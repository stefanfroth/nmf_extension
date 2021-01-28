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