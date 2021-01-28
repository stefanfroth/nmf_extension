import pytest

@pytest.fixture
def test_data():
    import numpy as np

    # Set numpy random seed
    np.random.seed(598)

    # Create the user-feature matrix P and the item-feature matrix Q
    P = np.random.randint(0, 17, (1000, 4))/10
    Q = np.random.randint(0, 17, (4, 1000))/10

    # Construct R and save it to
    R = np.matmul(P, Q)

    # Randomly create some missing values
    R_missing = R
    mask = np.random.randint(0, 1000, (1000, 1000))
    R_missing[mask<200] = np.nan

    return (R, R_missing)