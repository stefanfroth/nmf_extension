# %%
'''
Extend sklearns NMF algorithm to perform work with missing values.
'''
import numbers
import warnings

import pandas as pd
import numpy as np
from sklearn.decomposition._nmf import _check_init, _fit_multiplicative_update, _compute_regularization, _check_string_param, _beta_loss_to_float, _initialize_nmf, _update_coordinate_descent, norm,  _fit_multiplicative_update, _beta_divergence
from sklearn.decomposition._cdnmf_fast import _update_cdnmf_fast
from sklearn.decomposition import NMF
from sklearn.utils import check_array, _deprecate_positional_args, check_X_y, check_random_state
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot
from sklearn.utils.validation import check_non_negative, check_is_fitted
from sklearn.exceptions import ConvergenceWarning
from sklearn._config import config_context

np.random.seed(10)

## TODO: alpha here is the learning rate. alpha in sklearn is the regularization strength. Change that!
def custom_non_negative_factorization(X, y=None, W=None, H=None, n_iter_=200, alpha=0.0001):
        try:
            if not W:
                W = initialize_user_feature_matrix(X.shape[0], H.shape[0])
        except:
            pass
        # H = initialize_item_feature_matrix(X.shape[1], self.n_components)
        Rhat = np.dot(W, H)
        errors = Rhat - X
        for i in range(n_iter_):
            W = update_user_feature_matrix(W, H, alpha, errors)
            H = update_item_feature_matrix(W, H, alpha, errors)
            Rhat = np.matmul(W, H)
            errors = Rhat - X
            # print(f'We are in iteration {i}')
            # print(f'The mse is {round(np.nansum(errors**2), 2)}')
        return W, H, n_iter_

def _compute_regularization(alpha, l1_ratio, regularization):
    """Compute L1 and L2 regularization coefficients for W and H."""
    alpha_H = 0.
    alpha_W = 0.
    if regularization in ('both', 'components'):
        alpha_H = float(alpha)
    if regularization in ('both', 'transformation'):
        alpha_W = float(alpha)

    l1_reg_W = alpha_W * l1_ratio
    l1_reg_H = alpha_H * l1_ratio
    l2_reg_W = alpha_W * (1. - l1_ratio)
    l2_reg_H = alpha_H * (1. - l1_ratio)
    return l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H

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
    return np.random.randint(1, 3, size=(nr_of_users, n_components)).astype(float)

def initialize_item_feature_matrix(nr_of_items, n_components):
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
    return np.random.randint(1, 3, size=(n_components, nr_of_items)).astype(float)


def update_user_feature_matrix(user_feature_matrix, item_feature_matrix, alpha, errors):
    # calculate the updates for user_feature_matrix

    ## Numpy version reduces runtime by roughly 3/4 only for user_feature_matrix
    for user in range(user_feature_matrix.shape[0]):
        for feature in range(user_feature_matrix.shape[1]):
            update = 0
            for i, error in enumerate(errors[user]):
                if not np.isnan(error):
                    update -= 2*alpha*error*item_feature_matrix[feature,i]
            user_feature_matrix[user, feature] = np.maximum(user_feature_matrix[user, feature]+update, 0)
            # print(f'The new user_feature_matrix is {user_feature_matrix}')
    return user_feature_matrix

    # user_feature_matrix = pd.DataFrame(user_feature_matrix)
    # item_feature_matrix = pd.DataFrame(item_feature_matrix)
    # errors = pd.DataFrame(errors)
    # for user in user_feature_matrix.index:
    #     for feature in user_feature_matrix.columns:
    #         update = 0
    #         old = user_feature_matrix.at[user, feature]
    #         for i, error in enumerate(errors.iloc[user]):
    #             if not pd.isna(error):
    #                 update -= 2*alpha*error*item_feature_matrix.loc[feature].iat[i]
    #         user_feature_matrix.at[user, feature] = old+update
    # return user_feature_matrix.values


def update_item_feature_matrix(user_feature_matrix, item_feature_matrix, alpha, errors):
    # calculate the updates for user_feature_matrix
    ## Introducing Numpy for both update functions reduces the runtime by 90%
    for feature in range(item_feature_matrix.shape[0]):
        for item in range(item_feature_matrix.shape[1]):
            update = 0
            for i, error in enumerate(errors[:,item]):
                if not np.isnan(error):
                    update -= 2*alpha*error*user_feature_matrix[i,feature]
            item_feature_matrix[feature, item] = np.maximum(item_feature_matrix[feature, item]+update, 0) 
            # print(f'The new item_feature_matrix is {item_feature_matrix}')
    return item_feature_matrix


    # user_feature_matrix = pd.DataFrame(user_feature_matrix)
    # item_feature_matrix = pd.DataFrame(item_feature_matrix)
    # errors = pd.DataFrame(errors)
    # for feature in item_feature_matrix.index:
    #     for item in item_feature_matrix.columns:
    #         update = 0
    #         old = item_feature_matrix.at[feature, item]
    #         for i, error in enumerate(errors.iloc[:,item]):
    #             if not pd.isna(error):
    #                 update -= 2*alpha*error*user_feature_matrix.loc[:,feature].iat[i]
    #         item_feature_matrix.at[feature, item] = old+update
    # return item_feature_matrix.values


def _update_coordinate_descent(X, W, Ht, l1_reg, l2_reg, shuffle,
                               random_state):
    """Helper function for _fit_coordinate_descent
    Update W to minimize the objective function, iterating once over all
    coordinates. By symmetry, to update H, one can call
    _update_coordinate_descent(X.T, Ht, W, ...)
    """
    n_components = Ht.shape[1]

    HHt = np.dot(Ht.T, Ht)
    ## Added: Make the missing values zeros
    
    # TODO: This does not work. It has the same effect as imputing the value with 0
    X[np.isnan(X)] = 0
    XHt = safe_sparse_dot(X, Ht)
    #print(f'X is {X}')
    #print(f'Ht is {Ht}')
    #print(f'XHt is {XHt}')
    # TODO: https://stackoverflow.com/questions/57765137/nansum-only-if-at-least-one-value-is-not-nan-numpy
    # np.lib.nanfunctions._replace_nan(arr,value)
    # XHt[np.isnan(XHt)] = np.dot(np.ones(shape=(1, X.shape[1])), Ht)[0]

    # L2 regularization corresponds to increase of the diagonal of HHt
    if l2_reg != 0.:
        # adds l2_reg only on the diagonal
        HHt.flat[::n_components + 1] += l2_reg
    # L1 regularization corresponds to decrease of each element of XHt
    if l1_reg != 0.:
        XHt -= l1_reg

    if shuffle:
        permutation = random_state.permutation(n_components)
    else:
        permutation = np.arange(n_components)
    # The following seems to be required on 64-bit Windows w/ Python 3.5.
    permutation = np.asarray(permutation, dtype=np.intp)

    # Ultimately returns an updated to a coordinate in 
    return _update_cdnmf_fast(W, HHt, XHt, permutation)


# TODO: get rid of default random state
def initialize_wh(X, n_components, n_samples, n_features, random_state=42):
    '''
    Initializes component matrices
    '''
    avg = np.sqrt(np.nanmean(X) / n_components)
    rng = check_random_state(random_state)
    H = avg * rng.randn(n_components, n_features).astype(X.dtype,
                                                        copy=False)
    W = avg * rng.randn(n_samples, n_components).astype(X.dtype,
                                                        copy=False)
    np.abs(H, out=H)
    np.abs(W, out=W)
    return W, H

def _fit_coordinate_descent(X, W, H, tol=1e-4, max_iter=200, l1_reg_W=0,
                            l1_reg_H=0, l2_reg_W=0, l2_reg_H=0, update_H=True,
                            verbose=0, shuffle=False, random_state=None):
    """Compute Non-negative Matrix Factorization (NMF) with Coordinate Descent
    The objective function is minimized with an alternating minimization of W
    and H. Each minimization is done with a cyclic (up to a permutation of the
    features) Coordinate Descent.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Constant matrix.
    W : array-like, shape (n_samples, n_components)
        Initial guess for the solution.
    H : array-like, shape (n_components, n_features)
        Initial guess for the solution.
    tol : float, default: 1e-4
        Tolerance of the stopping condition.
    max_iter : integer, default: 200
        Maximum number of iterations before timing out.
    l1_reg_W : double, default: 0.
        L1 regularization parameter for W.
    l1_reg_H : double, default: 0.
        L1 regularization parameter for H.
    l2_reg_W : double, default: 0.
        L2 regularization parameter for W.
    l2_reg_H : double, default: 0.
        L2 regularization parameter for H.
    update_H : boolean, default: True
        Set to True, both W and H will be estimated from initial guesses.
        Set to False, only W will be estimated.
    verbose : integer, default: 0
        The verbosity level.
    shuffle : boolean, default: False
        If true, randomize the order of coordinates in the CD solver.
    random_state : int, RandomState instance, default=None
        Used to randomize the coordinates in the CD solver, when
        ``shuffle`` is set to ``True``. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.
    Returns
    -------
    W : array-like, shape (n_samples, n_components)
        Solution to the non-negative least squares problem.
    H : array-like, shape (n_components, n_features)
        Solution to the non-negative least squares problem.
    n_iter : int
        The number of iterations done by the algorithm.
    References
    ----------
    Cichocki, Andrzej, and Phan, Anh-Huy. "Fast local algorithms for
    large scale nonnegative matrix and tensor factorizations."
    IEICE transactions on fundamentals of electronics, communications and
    computer sciences 92.3: 708-721, 2009.
    """
    # so W and Ht are both in C order in memory
    Ht = check_array(H.T, order='C')
    X = check_array(X, force_all_finite=False, accept_sparse='csr')

    rng = check_random_state(random_state)

    for n_iter in range(1, max_iter + 1):
        violation = 0.

        # Update W
        violation += _update_coordinate_descent(X, W, Ht, l1_reg_W,
                                                l2_reg_W, shuffle, rng)
        # Update H
        if update_H:
            violation += _update_coordinate_descent(X.T, Ht, W, l1_reg_H,
                                                    l2_reg_H, shuffle, rng)

        if n_iter == 1:
            violation_init = violation

        if violation_init == 0:
            break

        if verbose:
            print("violation:", violation / violation_init)

        if violation / violation_init <= tol:
            if verbose:
                print("Converged at iteration", n_iter + 1)
            break

    return W, Ht.T, n_iter

@_deprecate_positional_args
def non_negative_factorization(X, W=None, H=None, n_components=None, *,
                               init=None, update_H=True, solver='cd',
                               beta_loss='frobenius', tol=1e-4,
                               max_iter=200, alpha=0., l1_ratio=0.,
                               regularization=None, random_state=42, # TODO: get rid of default random_state
                               verbose=0, shuffle=False):
    r"""Compute Non-negative Matrix Factorization (NMF)
    Find two non-negative matrices (W, H) whose product approximates the non-
    negative matrix X. This factorization can be used for example for
    dimensionality reduction, source separation or topic extraction.
    The objective function is::
        0.5 * ||X - WH||_Fro^2
        + alpha * l1_ratio * ||vec(W)||_1
        + alpha * l1_ratio * ||vec(H)||_1
        + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2
        + 0.5 * alpha * (1 - l1_ratio) * ||H||_Fro^2
    Where::
        ||A||_Fro^2 = \sum_{i,j} A_{ij}^2 (Frobenius norm)
        ||vec(A)||_1 = \sum_{i,j} abs(A_{ij}) (Elementwise L1 norm)
    For multiplicative-update ('mu') solver, the Frobenius norm
    (0.5 * ||X - WH||_Fro^2) can be changed into another beta-divergence loss,
    by changing the beta_loss parameter.
    The objective function is minimized with an alternating minimization of W
    and H. If H is given and update_H=False, it solves for W only.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Constant matrix.
    W : array-like, shape (n_samples, n_components)
        If init='custom', it is used as initial guess for the solution.
    H : array-like, shape (n_components, n_features)
        If init='custom', it is used as initial guess for the solution.
        If update_H=False, it is used as a constant, to solve for W only.
    n_components : integer
        Number of components, if n_components is not set all features
        are kept.
    init : None | 'random' | 'nndsvd' | 'nndsvda' | 'nndsvdar' | 'custom'
        Method used to initialize the procedure.
        Default: None.
        Valid options:
        - None: 'nndsvd' if n_components < n_features, otherwise 'random'.
        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)
        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)
        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)
        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)
        - 'custom': use custom matrices W and H
        .. versionchanged:: 0.23
            The default value of `init` changed from 'random' to None in 0.23.
    update_H : boolean, default: True
        Set to True, both W and H will be estimated from initial guesses.
        Set to False, only W will be estimated.
    solver : 'cd' | 'mu'
        Numerical solver to use:
        - 'cd' is a Coordinate Descent solver that uses Fast Hierarchical
            Alternating Least Squares (Fast HALS).
        - 'mu' is a Multiplicative Update solver.
        .. versionadded:: 0.17
           Coordinate Descent solver.
        .. versionadded:: 0.19
           Multiplicative Update solver.
    beta_loss : float or string, default 'frobenius'
        String must be in {'frobenius', 'kullback-leibler', 'itakura-saito'}.
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
        matrix X cannot contain zeros. Used only in 'mu' solver.
        .. versionadded:: 0.19
    tol : float, default: 1e-4
        Tolerance of the stopping condition.
    max_iter : integer, default: 200
        Maximum number of iterations before timing out.
    alpha : double, default: 0.
        Constant that multiplies the regularization terms.
    l1_ratio : double, default: 0.
        The regularization mixing parameter, with 0 <= l1_ratio <= 1.
        For l1_ratio = 0 the penalty is an elementwise L2 penalty
        (aka Frobenius Norm).
        For l1_ratio = 1 it is an elementwise L1 penalty.
        For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
    regularization : 'both' | 'components' | 'transformation' | None
        Select whether the regularization affects the components (H), the
        transformation (W), both or none of them.
    random_state : int, RandomState instance, default=None
        Used for NMF initialisation (when ``init`` == 'nndsvdar' or
        'random'), and in Coordinate Descent. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.
    verbose : integer, default: 0
        The verbosity level.
    shuffle : boolean, default: False
        If true, randomize the order of coordinates in the CD solver.
    Returns
    -------
    W : array-like, shape (n_samples, n_components)
        Solution to the non-negative least squares problem.
    H : array-like, shape (n_components, n_features)
        Solution to the non-negative least squares problem.
    n_iter : int
        Actual number of iterations.
    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> from sklearn.decomposition import non_negative_factorization
    >>> W, H, n_iter = non_negative_factorization(X, n_components=2,
    ... init='random', random_state=0)
    References
    ----------
    Cichocki, Andrzej, and P. H. A. N. Anh-Huy. "Fast local algorithms for
    large scale nonnegative matrix and tensor factorizations."
    IEICE transactions on fundamentals of electronics, communications and
    computer sciences 92.3: 708-721, 2009.
    Fevotte, C., & Idier, J. (2011). Algorithms for nonnegative matrix
    factorization with the beta-divergence. Neural Computation, 23(9).
    """
    X = check_array(X, force_all_finite=False, accept_sparse=('csr', 'csc'),
                    dtype=[np.float64, np.float32])
    check_non_negative(X, "NMF (input X)")
    beta_loss = _check_string_param(solver, regularization, beta_loss, init)

    if X.min() == 0 and beta_loss <= 0:
        raise ValueError("When beta_loss <= 0 and X contains zeros, "
                         "the solver may diverge. Please add small values to "
                         "X, or use a positive beta_loss.")

    n_samples, n_features = X.shape
    if n_components is None:
        n_components = n_features

    if not isinstance(n_components, numbers.Integral) or n_components <= 0:
        raise ValueError("Number of components must be a positive integer;"
                         " got (n_components=%r)" % n_components)
    if not isinstance(max_iter, numbers.Integral) or max_iter < 0:
        raise ValueError("Maximum number of iterations must be a positive "
                         "integer; got (max_iter=%r)" % max_iter)
    if not isinstance(tol, numbers.Number) or tol < 0:
        raise ValueError("Tolerance for stopping criteria must be "
                         "positive; got (tol=%r)" % tol)

    # check W and H, or initialize them
    if init == 'custom' and update_H:
        _check_init(H, (n_components, n_features), "NMF (input H)")
        _check_init(W, (n_samples, n_components), "NMF (input W)")
        if H.dtype != X.dtype or W.dtype != X.dtype:
            raise TypeError("H and W should have the same dtype as X. Got "
                            "H.dtype = {} and W.dtype = {}."
                            .format(H.dtype, W.dtype))
    elif not update_H:
        _check_init(H, (n_components, n_features), "NMF (input H)")
        if H.dtype != X.dtype:
            raise TypeError("H should have the same dtype as X. Got H.dtype = "
                            "{}.".format(H.dtype))
        # 'mu' solver should not be initialized by zeros
        if solver == 'mu':
            avg = np.sqrt(X.mean() / n_components)
            W = np.full((n_samples, n_components), avg, dtype=X.dtype)
        else:
            W = np.zeros((n_samples, n_components), dtype=X.dtype)
    else:
        W, H = _initialize_nmf(X, n_components, init=init,
                               random_state=random_state)

    l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H = _compute_regularization(
        alpha, l1_ratio, regularization)

    if solver == 'cd':
        W, H, n_iter = _fit_coordinate_descent(X, W, H, tol, max_iter,
                                               l1_reg_W, l1_reg_H,
                                               l2_reg_W, l2_reg_H,
                                               update_H=update_H,
                                               verbose=verbose,
                                               shuffle=shuffle,
                                               random_state=random_state)
    elif solver == 'mu':
        W, H, n_iter = _fit_multiplicative_update(X, W, H, beta_loss, max_iter,
                                                  tol, l1_reg_W, l1_reg_H,
                                                  l2_reg_W, l2_reg_H, update_H,
                                                  verbose)

    else:
        raise ValueError("Invalid solver parameter '%s'." % solver)

    if n_iter == max_iter and tol > 0:
        warnings.warn("Maximum number of iterations %d reached. Increase it to"
                      " improve convergence." % max_iter, ConvergenceWarning)

    return W, H, n_iter


# class CustomBaseEstimator(BaseEstimator):
#     '''
#     Rewrite the BaseEstimator so that it allows nan values.
#     '''

    
class CustomNMF(NMF):
    '''
    CustomNMF class that allows for non-negative input
    '''
    # Set the initialization to random to avoid problems with NaNs (eg. for SVD)

    def _validate_data(self, X, y=None, reset=True,
                       validate_separately=False, **check_params):
        """Validate input data and set or check the `n_features_in_` attribute.
        Parameters
        ----------
        X : {array-like, sparse matrix, dataframe} of shape \
                (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,), default=None
            The targets. If None, `check_array` is called on `X` and
            `check_X_y` is called otherwise.
        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.
        validate_separately : False or tuple of dicts, default=False
            Only used if y is not None.
            If False, call validate_X_y(). Else, it must be a tuple of kwargs
            to be used for calling check_array() on X and y respectively.
        **check_params : kwargs
            Parameters passed to :func:`sklearn.utils.check_array` or
            :func:`sklearn.utils.check_X_y`. Ignored if validate_separately
            is not False.
        Returns
        -------
        out : {ndarray, sparse matrix} or tuple of these
            The validated input. A tuple is returned if `y` is not None.
        """

        if y is None:
            if self._get_tags()['requires_y']:
                raise ValueError(
                    f"This {self.__class__.__name__} estimator "
                    f"requires y to be passed, but the target y is None."
                )
            # Allow-nan for X
            X = check_array(X, force_all_finite=False, **check_params)
            out = X
        else:
            if validate_separately:
                # We need this because some estimators validate X and y
                # separately, and in general, separately calling check_array()
                # on X and y isn't equivalent to just calling check_X_y()
                # :(
                check_X_params, check_y_params = validate_separately
                X = check_array(X, **check_X_params)
                y = check_array(y, **check_y_params)
            else:
                X, y = check_X_y(X, y, **check_params)
            out = X, y

        if check_params.get('ensure_2d', True):
            self._check_n_features(X, reset=reset)

        return out
    
    
    def fit_transform(self, X, y=None, W=None, H=None):
        """Learn a NMF model for the data X and returns the transformed data.
        This is more efficient than calling fit followed by transform.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix to be decomposed
        y : Ignored
        W : array-like, shape (n_samples, n_components)
            If init='custom', it is used as initial guess for the solution.
        H : array-like, shape (n_components, n_features)
            If init='custom', it is used as initial guess for the solution.
        Returns
        -------
        W : array, shape (n_samples, n_components)
            Transformed data.
        """
        # TODO: How is this handled in sklearn in general?
        # Convert to numpy array if a pd.DataFrame is passed
        if isinstance(X, pd.DataFrame):
            X = X.values

        # TODO: find right spot for this
        # Set the initializer to custom to avoid problems with missing values
        self.init = 'custom'
        # Randomly initialize W and X
        W, H = initialize_wh(X, self.n_components, len(X), X.shape[1])

        X = self._validate_data(X, accept_sparse=('csr', 'csc'),
                                dtype=[np.float64, np.float32])

        ## TODO: Find a proper solution for an adaptive learning rate
        #LEARNING_RATE=0.5*1/10**len(str(X.shape[0])) + 0.5*1/10**len(str(X.shape[1]))
        LEARNING_RATE=0.0001

        W, H, n_iter_ = custom_non_negative_factorization(X=X, W=W, H=H, 
                            n_iter_=self.max_iter, alpha=LEARNING_RATE)
        # non_negative_factorization(
        #     X=X, W=W, H=H, n_components=self.n_components, init=self.init,
        #     update_H=True, solver=self.solver, beta_loss=self.beta_loss,
        #     tol=self.tol, max_iter=self.max_iter, alpha=self.alpha,
        #     l1_ratio=self.l1_ratio, regularization='both',
        #     random_state=self.random_state, verbose=self.verbose,
        #     shuffle=self.shuffle)

        self.reconstruction_err_ = _beta_divergence(X, W, H, self.beta_loss,
                                                    square_root=True)

        self.n_components_ = H.shape[0]
        self.components_ = H
        self.n_iter_ = n_iter_

        return W


    def transform(self, X):
        """Transform the data X according to the fitted NMF model.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be transformed by the model.
        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=('csr', 'csc'),
                                dtype=[np.float64, np.float32],
                                reset=False)

        ## TODO: Find a proper solution for an adaptive learning rate
        # LEARNING_RATE=0.5*1/10**len(str(X.shape[0])) + 0.5*1/10**len(str(X.shape[1]))
        LEARNING_RATE=0.0001

        with config_context(assume_finite=True):
            W, _, n_iter_ = custom_non_negative_factorization(X=X, W=None, H=self.components_, 
                            n_iter_=self.max_iter, alpha=LEARNING_RATE)
            # non_negative_factorization(
            #     X=X, W=None, H=self.components_,
            #     n_components=self.n_components_,
            #     init=self.init, update_H=False, solver=self.solver,
            #     beta_loss=self.beta_loss, tol=self.tol, max_iter=self.max_iter,
            #     alpha=self.alpha, l1_ratio=self.l1_ratio,
            #     regularization=None, # TODO: what to do about this? self.regularization,
            #     random_state=self.random_state,
            #     verbose=self.verbose, shuffle=self.shuffle)

        return W