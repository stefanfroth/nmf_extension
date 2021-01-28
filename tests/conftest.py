import pytest
from nmf_extension.nmf import CustomNMF

# Set number of components
N_COMPONENTS = 4

## TODO: Check whether it is faster to load from a file or to create new each time
## TODO: Parametrize the fixture to change amount of data missing
@pytest.fixture
def test_data():
    import numpy as np

    # Set numpy random seed
    ## TODO: Test result for test_better depend on the random seed and the amount of missing data
    np.random.seed(598)
    #np.random.seed(2)

    # Create the user-feature matrix P and the item-feature matrix Q
    P = np.random.randint(0, 17, (1000, N_COMPONENTS))/10
    Q = np.random.randint(0, 17, (N_COMPONENTS, 1000))/10

    # Construct R and save it to
    R = np.matmul(P, Q)

    # Randomly create some missing values
    R_missing = R
    mask = np.random.randint(0, 1000, (1000, 1000))
    R_missing[mask<200] = np.nan

    return (R, R_missing)

@pytest.fixture
def test_model():
    '''Creates an instance of the CustomNMF'''
    return CustomNMF(n_components=N_COMPONENTS)