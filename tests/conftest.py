import numpy as np
import pytest
from sklearn.metrics import mean_squared_error
from nmf_extension.nmf import CustomNMF

# Set number of components
@pytest.fixture(scope='module')
def components():
    return 4

# Set the maximum number of iterations
@pytest.fixture(scope='module')
def iterations():
    return 200

# Set number of users
@pytest.fixture(scope='module')
def n_users():
    return 300

# Set number of items
@pytest.fixture(scope='module')
def n_items():
    return 300

## TODO: Check whether it is faster to load from a file or to create new each time
## TODO: Parametrize the fixture to change amount of data missing
## TODO: Make n and m a fixture itself
@pytest.fixture(scope='module')
def test_data(components, n_users, n_items):
    import numpy as np

    # Set numpy random seed
    ## TODO: Test result for test_better depend on the random seed and the amount of missing data
    np.random.seed(598)
    #np.random.seed(2)

    # Create the user-feature matrix P and the item-feature matrix Q
    P = np.random.randint(0, 17, (n_users, components))/10
    Q = np.random.randint(0, 17, (components, n_items))/10

    # Construct R and save it to
    R = np.matmul(P, Q)

    # Randomly create some missing values
    R_missing = R
    mask = np.random.randint(0, 1000, (n_users, n_items))
    R_missing[mask<200] = np.nan

    return (R, R_missing)

@pytest.fixture(scope='module')
def test_model(components, iterations, test_data):
    '''Creates an instance of the CustomNMF'''
    R_missing = test_data[1]
    nmf = CustomNMF(n_components=components, max_iter=iterations)
    nmf.fit(R_missing)
    return nmf

@pytest.fixture(scope='module')
def p_hat(test_model, test_data):
    '''Create R_hat'''
    return test_model.transform(test_data[1])

@pytest.fixture(scope='module')
def r_hat(test_model, p_hat):
    '''Create r_hat'''
    return np.matmul(p_hat, test_model.components_)

@pytest.fixture(scope='module')
def mse_nmf(test_data, r_hat):
    '''Calculate the nmf for the test model'''
    R_missing = test_data[0]
    mse_custom = mean_squared_error(R_missing[~np.isnan(R_missing)], r_hat[~np.isnan(R_missing)])
    return mse_custom
