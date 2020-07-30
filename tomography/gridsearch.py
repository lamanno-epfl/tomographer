import numpy as np
import pandas as pd
from .core import build_Design_Matrix
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
from typing import *
# from skimage import color


def zero_shifting(A: np.ndarray, n: int) -> np.ndarray:
    """ Zero pad (from the left) or index A to make it fir a certain lenght
    """
    if n > 0:
        return np.r_[(0,) * n, A]
    elif n < 0:
        return A[-n:]
    else:
        return A


def equilibrate(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Expand the shorter of two array by zero padding
    """
    le = max(len(a), len(b))
    tempa, tempb = np.zeros((le,)), np.zeros((le,)) 
    tempa[:len(a)] = a
    tempb[:len(b)] = b
    return tempa, tempb


def mixed_interpolator(xi: np.ndarray, p: np.ndarray) -> np.ndarray:
    """ Perform a mixed cubic/linear interporlation

    Args
    ----
    xi: np.ndarray
        the x position mesured/not-excluded
    p: np.ndarray
        the detection level

    Returns
    -------
    interpolated_p: np.ndarray
        the interpoleted values of p

    """
    full_x = np.arange(xi[0], xi[-1] + 1)
    interpolator_c = interp1d(xi, p, kind='cubic', fill_value=0, bounds_error=False)
    interpolator_l = interp1d(xi, p, kind='linear', fill_value=0, bounds_error=False)
    return 0.15 * np.clip(interpolator_c(full_x), a_min=0, a_max=2*np.max(p)) +\
    0.35 * np.clip(interpolator_l(full_x), a_min=0, a_max=2*np.max(p))  +\
    0.5 * np.clip(interpolator_l(full_x), a_min=0, a_max=2 * np.max(p)) * np.array([(i in xi) for i in full_x])


def objective_afterscaling(angle: float, width: float, experimental: np.ndarray, reference_img: np.ndarray, mask: np.ndarray, objective: Callable=None) -> Tuple[float, float]:
    """ Find the optimal vertical scaling given the angle and the width.
    This solves the problem of different scaling between the reference mask (arbitrary) and the mesured expression levels (reads/UMI).

    Args
    ----
    angle : float
        The angle of projection in radiant
    width: float
        The thikness of the strips in pixels
    experimental: np.ndarray
        The mesured value for the variable
    reference_img: np.ndarray 2d
        The reference image that prejected should give `experimental`
    mask: np.ndarray
        The mask for the tissue area
    objective: Callable default None
        objecting function that takes a scalar attribute
        if None: Residual Sum of Squares will be used.

    Returns
    -------
    obj_val: float
        The value of the objective function at the minimum
    scaling: float
        The 
    """
    Ex = experimental
    # Calculate the masked projection matrix
    D, projs_len = build_Design_Matrix(np.array([angle]), [width], 1. * (mask >= 0.2))
    # Project the reference using the masked projection matrix
    T = D.dot(reference_img.flat[:])
    # Adjust the lenghts of the vectors
    T, Ex = equilibrate(T, Ex)
    # Optimize scaling
    if objective is None:
        def objective(scale: float, X: np.ndarray, Y: np.ndarray) -> float:
            return np.sum((Y - (scale * X))**2)
    res = minimize_scalar(objective, args=(T, Ex))
    # Alternatives objectives
    # res = minimize(lambda x,A,B: np.sum(((x*A)-B)**2), [1,], args=(T,Ex))
    # res = minimize(lambda x,A,B: sum(abs((x*A)-B)), [1,], args=(T,Ex))
    # Calculate the residual sum of squares
    if res.success:
        obj_val = res.fun
        scaling = res.x
    else:
        obj_val = np.inf
        scaling = np.inf
    return obj_val, scaling


def gridsearch_allignment(grid_angles: Iterable, grid_wids: Iterable, grid_zeroshifts: Iterable, expvalues: np.ndarray, xi: np.ndarray, reference_img: np.ndarray, mask: np.ndarray, objective: Callable = None) -> pd.DataFrame:
    ''' Perform a grid search of the optimal allignment exploring all the possible combinations of angles, widths and shifts.
    It uses a reference image and the expression values and compares them using a objective function

    Args
    ----
    grid_angles: Iterable
        Iterable of the angles to try. The angle of projection in radiant
    grid_wids: Iterable
        Iterable of the widths to try.
        Width is the thikness of the strips in pixels
    grid_zeroshifts: Iterable
        Terable of the zeroshifts to try
    expvalues: np.ndarray
        The variable to allign
    xi: np.ndarray
        The indexes of the variable to allign
    reference_img: np.ndarray
        The reference image that is projected and alligned with `` expvalues
    mask: np.ndarray
        The mask used build the design matrix
    objective: Callable
        Function to minimize. Should take 3 arguments f(s,X,Y) where X is the measured variable Y is the reference after projecting `reference_img`

    Returns
    -------
    results: pd.Dataframe
        The results organized in a pandas Dataframe.
        with columns: "objective", "angle", "width", "scaling", "shift"

    '''
    list_results = []  # type : Tuple
    for angle_x in grid_angles:
        for width_x in grid_wids:
            for shift_x in grid_zeroshifts:
                expvalues_z = zero_shifting(mixed_interpolator(xi, expvalues), shift_x)
                obj_val, scaling = objective_afterscaling(angle_x, width_x, expvalues_z, reference_img, mask, objective)
                list_results.append([obj_val, angle_x, width_x, scaling, shift_x])
    return pd.DataFrame(data=np.array(list_results), columns=["objective", "angle", "width", "scaling", "shift"])
