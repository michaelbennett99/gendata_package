"""
Module containing functions for calculating covariances and covariance matrices.
"""

from typing import Union, Optional

import numpy as np

from numba import njit

@njit(fastmath=True)
def cov(row_i: np.ndarray, row_j: np.ndarray) -> float:
    """Compute the covariance between two 1D numpy arrays.

    :math:`cov(X, Y) = \\frac{1}{n} \\sum_{i=1}^{n} (X_i - \\bar{X})(Y_i - \\bar{Y})`

    :param row_i: Numpy array of length n_obs.
    :type row_i: np.ndarray

    :param row_j: Numpy array of length n_obs.
    :type row_j: np.ndarray

    :return: Covariance between row_i and row_j.
    :rtype: float

    :raises ValueError: If the arrays do not have the same length.
    """
    return np.dot(row_i, row_j) / row_i.shape[0]

@njit(fastmath=True)
def weighted_cov(
        row_i: np.ndarray, row_j: np.ndarray, weights: np.ndarray
    ) -> float:
    """Compute the weighted covariance between two 1D numpy arrays.

    :math:`cov(X, Y) = \\frac{1}{\\sum w} \\sum_{i=1}^{n} w_i (X_i - \\bar{X})(Y_i - \\bar{Y})`

    :param row_i: Numpy array of length n_obs.
    :type row_i: np.ndarray

    :param row_j: Numpy array of length n_obs.
    :type row_j: np.ndarray

    :param weights: Numpy array of length n_obs.
    :type weights: np.ndarray

    :return: Weighted covariance between row_i and row_j.
    :rtype: float
    """
    return np.dot(row_i, row_j * weights) / np.sum(weights)


@njit(fastmath=True)
def cov_na(row_i: np.ndarray, row_j: np.ndarray) -> tuple[float, int]:
    """Compute the covariance between two 1D numpy arrays with nan elements.

    :param row_i: 1D numpy array with n_i observations.
    :type row_i: np.ndarray

    :param row_j: 1D numpy array with n_j observations.
    :type row_j: np.ndarray

    :return: Covariance between row_i and row_j.
    :rtype: float
    """
    mult = row_i * row_j
    multsum = np.nansum(mult)
    n_intersection = (~np.isnan(mult)).sum()
    return multsum / n_intersection, n_intersection


@njit(fastmath=True)
def make_spwindow_cov(
        mat: np.ndarray,
        position: np.ndarray,
        window: Union[int, float],
        missing: Optional[np.ndarray] = None,
        tol: float = 0.001
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Makes a sparse covariance matrix, where all elements for which
    distance < window are set to zero.

    Variables must be ordered by their position.

    :param mat: MxN Array to make covariance matrix for. Assumed that variables
        are in rows and observations in columns.
    :type mat: np.ndarray

    :param position: A length M vector recording the position of each variable.
    :type position: np.ndarray

    :param window: Only calculate the covariance between 2 variables if the
        distance between them is less than this.
    :type window: Union[int, float]

    :param missing: A length M vector recording the missingness of each
        variable. Defaults to None, in which case missingness will be
        calculated.
    :type missing: Optional[np.ndarray]

    :param tol: Tolerance below which all values are set to 0. Defaults to
        0.001.
    :type tol: float

    :return: Covariance matrix in sparse ijv format.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, list]
    """
    n_var, n_obs = mat.shape

    if missing is None: # then calculate missingness array
        missing = (n_obs - np.count_nonzero(mat, axis=1)) != 0

    # This is a conservative estimate for the required size of the output
    max_len = n_var ** 2

    rows_remove = []

    ldm_i = np.empty(max_len, dtype=np.uint32)
    ldm_j = np.empty(max_len, dtype=np.uint32)
    ldm_val = np.empty(max_len, dtype=np.float32)

    counter = 0
    for i in range(n_var):
        i_counter = counter
        j_counter = 0

        i_len = 2 * (n_var - i) - 1
        i_ldm_i = np.empty(i_len, dtype=np.uint32)
        i_ldm_j = np.empty(i_len, dtype=np.uint32)
        i_ldm_val = np.empty(i_len, dtype=np.float32)
        for j in range(i, n_var):
            valid = np.abs(position[i] - position[j]) < window

            if not valid: # then no variables further away will be valid
                # Add this row to the matrix
                ldm_i[i_counter: counter] = i_ldm_i[0:j_counter]
                ldm_j[i_counter: counter] = i_ldm_j[0:j_counter]
                ldm_val[i_counter: counter] = i_ldm_val[0:j_counter]
                # Then break
                break

            else: # calculate covariances

                if missing[i] or missing[j]:
                    val, _ = cov_na(mat[i], mat[j])
                else:
                    val = cov(mat[i], mat[j])

                if i == j:
                    i_ldm_i[j_counter] = i
                    i_ldm_j[j_counter] = j
                    i_ldm_val[j_counter] = val
                    counter += 1
                    j_counter += 1
                else:
                    if val == 1: # If this is 1 we will have a deficient matrix
                        counter = i_counter
                        rows_remove.append(i)
                        break
                    if abs(val) < tol:
                        val = 0

                    i_ldm_i[j_counter] = j
                    i_ldm_j[j_counter] = i
                    i_ldm_val[j_counter] = val
                    counter += 1
                    j_counter += 1

                    i_ldm_i[j_counter] = i
                    i_ldm_j[j_counter] = j
                    i_ldm_val[j_counter] = val
                    counter += 1
                    j_counter += 1
        else:
            # Set elements in main array
            ldm_i[i_counter: counter] = i_ldm_i
            ldm_j[i_counter: counter] = i_ldm_j
            ldm_val[i_counter: counter] = i_ldm_val
    return ldm_i[0:counter], ldm_j[0:counter], ldm_val[0:counter], rows_remove


@njit(fastmath=True)
def make_window_cov(
        mat: np.ndarray,
        position: np.ndarray,
        window: Union[int, float],
        missing: Optional[np.ndarray] = None
    ) -> np.ndarray:
    """Makes a dense covariance matrix where all elements for which distance
    < window are set to zero.

    Variables must be ordered by their position.

    :param mat: MxN Array to make covariance matrix for. Assumed that variables
        are in rows and observations in columns.
    :type mat: np.ndarray

    :param position: A length M vector recording the position of each variable.
    :type position: np.ndarray

    :param window: Only calculate the covariance between 2 variables if the
        distance between them is less than this.
    :type window: Union[int, float]

    :param missing: A length M vector recording the missingness of each
        variable. Defaults to None, in which case missingness will be
        calculated.
    :type missing: Optional[np.ndarray]

    :return: Covariance matrix in dense format.
    :rtype: np.ndarray
    """
    n_var, n_obs = mat.shape

    if missing is None:
        missing = (n_obs - np.count_nonzero(mat, axis=1)) != 0

    # This is a conservative estimate for the required size of the output
    ldm = np.zeros((n_var, n_var), dtype=np.float32)
    for i in range(n_var):
        for j in range(i, n_var):
            valid = np.abs(position[i] - position[j]) < window
            if not valid: # then no variables further away will be valid
                break
            else: # calculate covariances
                if missing[i] or missing[j]:
                    val, _ = cov_na(mat[i], mat[j])
                else:
                    val = cov(mat[i], mat[j])
                ldm[i, j] = val
                if i != j:
                    ldm[j, i] = val
    return ldm

@njit(fastmath=True)
def make_cov(
        mat: np.ndarray,
        missing: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray, set[int]]:
    """Make a dense covariance matrix, allowing for missing data.

    :param mat: MxN array to make covariance matrix for. Assumed that variables
        are in rows and observations in columns.
    :type mat: np.ndarray

    :param missing: A length M vector recording the missingness of each
        variable. Defaults to None, in which case missingness will be
        calculated.
    :type missing: Optional[np.ndarray]

    :return: Covariance matrix in dense format.
    :rtype: np.ndarray
    """
    n_var, n_obs = mat.shape

    if missing is None:
        missing = np.sum(np.isnan(mat), axis=1) != 0

    # This is the required size of the output
    ldm = np.zeros((n_var, n_var), dtype=np.float32)
    non_missing = np.zeros((n_var, n_var), dtype=np.uint32)
    rows_remove = set()
    for i in range(n_var):
        for j in range(i, n_var):
            if missing[i] or missing[j]:
                val, nm = cov_na(mat[i], mat[j])
            else:
                val = cov(mat[i], mat[j])
                nm = n_obs
            ldm[i, j] = val
            non_missing[i, j] = nm
            if i != j:
                lin_dep = (
                    (val == 1)
                    and (i not in rows_remove)
                    and (j not in rows_remove)
                )
                if lin_dep: # If this is 1 we will have a rank deficient matrix
                    rows_remove.add(i)
                ldm[j, i] = val
                non_missing[j, i] = nm
    return ldm, non_missing, rows_remove


@njit(fastmath=True)
def make_weighted_cov(
        mat: np.ndarray,
        weights: np.ndarray,
        missing: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray, set[int]]:
    """Make a dense covariance matrix, allowing for missing data.

    :param mat: MxN array to make covariance matrix for. Assumed that variables
        are in rows and observations in columns.
    :type mat: np.ndarray

    :param weights: A length N vector of weights for each observation.
    :type weights: np.ndarray

    :param missing: A length M vector recording the missingness of each
        variable. Defaults to None, in which case missingness will be
        calculated.
    :type missing: Optional[np.ndarray]

    :return: Covariance matrix in dense format.
    :rtype: np.ndarray
    """
    n_var, n_obs = mat.shape

    if missing is None:
        missing = np.sum(np.isnan(mat), axis=1) != 0

    if missing.any():
        raise ValueError("Missing data not supported for weighted covariance.")

    # This is the required size of the output
    ldm = np.zeros((n_var, n_var), dtype=np.float32)
    non_missing = np.zeros((n_var, n_var), dtype=np.uint32)
    rows_remove = set()
    for i in range(n_var):
        for j in range(i, n_var):
            val = weighted_cov(mat[i], mat[j], weights)
            ldm[i, j] = val
            non_missing[i, j] = n_obs
            if i != j:
                lin_dep = (
                    (val == 1)
                    and (i not in rows_remove)
                    and (j not in rows_remove)
                )
                if lin_dep: # If this is 1 we will have a rank deficient matrix
                    rows_remove.add(i)
                ldm[j, i] = val
                non_missing[j, i] = n_obs
    return ldm, non_missing, rows_remove
