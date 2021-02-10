import pytest
from nmf_extension.nmf import CustomNMF

# Set number of components
@pytest.fixture()
def components():
    return 4

# Set the maximum number of iterations
@pytest.fixture()
def iterations():
    return 1000

## TODO: Check whether it is faster to load from a file or to create new each time
## TODO: Parametrize the fixture to change amount of data missing
@pytest.fixture
def test_data(components, n=10, m=10):
    import numpy as np

    # Set numpy random seed
    ## TODO: Test result for test_better depend on the random seed and the amount of missing data
    np.random.seed(598)
    #np.random.seed(2)

    # Create the user-feature matrix P and the item-feature matrix Q
    P = np.random.randint(0, 17, (n, components))/10
    Q = np.random.randint(0, 17, (components, m))/10

    # Construct R and save it to
    R = np.matmul(P, Q)

    # Randomly create some missing values
    R_missing = R
    mask = np.random.randint(0, 1000, (n, m))
    R_missing[mask<200] = np.nan

    return (R, R_missing)

@pytest.fixture
def test_model(components, iterations):
    '''Creates an instance of the CustomNMF'''
    return CustomNMF(n_components=components, max_iter=iterations)