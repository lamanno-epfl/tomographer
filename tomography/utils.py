import numpy as np
import pandas as pd
from skimage.color import gray2rgb, rgb2hsv, hsv2rgb
from scipy.ndimage import gaussian_filter
from scipy.integrate import simps
from scipy.interpolate import interp1d
from typing import *
from .core import prepare_regression_symmetry
from .optimization import ReconstructorFastScipyNB


def pad_to_cube(ndarray: np.ndarray) -> np.ndarray:
    d0, d1, d2 = d = ndarray.shape
    p0, p1, p2 = p = (2 * np.ceil(max(d) / 2.).astype(int),) * 3
    padded = np.zeros(p, dtype=ndarray.dtype)
    s0, s1, s2 = starts = np.ceil((np.array(p) - np.array(d)) / 2).astype(int)
    padded[s0:s0 + d0,
           s1:s1 + d1,
           s2:s2 + d2] = ndarray
    return padded


def pad_to_square(ndarray: np.ndarray) -> np.ndarray:
    d0, d1 = d = ndarray.shape
    p0, p1 = p = (2 * np.ceil(max(d) / 2.).astype(int),) * 2
    padded = np.zeros(p, dtype=ndarray.dtype)
    s0, s1 = starts = np.ceil((np.array(p) - np.array(d)) / 2).astype(int)
    padded[s0:s0 + d0,
           s1:s1 + d1] = ndarray
    return padded


def shift_simmetrize(A: np.ndarray, shift: int) -> np.ndarray:
    X = np.zeros(A.shape)
    X[:, :, shift:] = A[:, :, :-shift]
    X[:, :, X.shape[-1] // 2:] = X[:, :, :X.shape[-1] // 2][:, :, ::-1]
    return X


def get_x(df: pd.DataFrame) -> np.ndarray:
    """
    Args
    ----

    df : pd.Dataframe
        Pandas dataframe that contains as columns formatted as follow
        angle{degrees}_{plate_num}_{well_num}_x{pos}

    Returns
    -------
    pos: np.ndarray
        the position on the projection axis ( orthogonal to the cutting angle)
        where the orientation of the axis reflects the order the strips have been cut

    """
    return np.array([float(i.split('x')[-1]) for i in df.columns])


def get_plate(df: pd.DataFrame) -> np.ndarray:
    """Extract for each well the 96 well plate it comes from

    Args
    ----

    df : pd.Dataframe
        Pandas dataframe that contains as columns formatted as follow
        angle{degrees}_{plate_num}_{well_num}_x{pos}

    Returns
    -------
    plate_num: np.ndarray
        the plate id of each sample

    """
    return np.array([int(i.split('_')[1]) for i in df.columns])


def colorize(image: np.ndarray, hue: float, saturation: float=1) -> np.ndarray:
    """ Add color of the given hue to an RGB image.

    By default, set the saturation to 1 so that the colors pop!

    Args
    ----
    image: np.ndarray
        origina image in RGB
    hue: float
        the hue we want to set
    saturation: float
        the saturation we want to set

    Returns
    -------
    new_image:
        the new image after the transformation

    """
    hsv = rgb2hsv(image)
    hsv[:, :, 1] = saturation
    hsv[:, :, 0] = hue
    return hsv2rgb(hsv)


def normalize_AUC(data: Dict[str, pd.DataFrame], return_scaling_factors: bool=False) -> Union[Tuple, Dict]:
    ''' Normalize the datasets so that the area under the curve is the same in all projections
    and equal to their average.

    Args
    ----
        data: Dict[str, pd.Dataframe]
            A dictionary containing the projections datasets to normalize. Every dataframe is assumed to be
            not filterd (e.g. containing all the genes) and of shape (NGenes,NProjections)

    Returns
    -------
    data : Dict[str, pd.Dataframe]
        the dictionary containing the different projections normalized
    sclaing_factors: optional
        if return_scaling_factors is True it returns the scaling factors that have been used for the normalization

    Notes
    -----
    It uses sympson integration instead of just summing to avoid error because of the removed bad samples.
    This assumes that dx is the same, even if later this will be slightly changed to best fit the projections.

    '''
    integrals = pd.Series()
    for name_angle in data.keys():
        x = get_x(data[name_angle])
        y = data[name_angle].sum(0)
        integrals[name_angle] = simps(y, x)
    scaling_factors = np.mean(integrals) / integrals
    data_norm = {}
    for name_angle in data.keys():
        data_norm[name_angle] = data[name_angle] * scaling_factors[name_angle]
    if return_scaling_factors:
        return data_norm, scaling_factors
    else:
        return data_norm


def mixed_interpolator2(xi: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    not_provided: np.ndarray
        boolean array with the values that have been fill markerd as True

    """
    full_x = np.arange(xi[0], xi[-1] + 1)
    interpolator_c = interp1d(xi, p, kind='cubic', fill_value=0, bounds_error=False)
    interpolator_l = interp1d(xi, p, kind='linear', fill_value=0, bounds_error=False)
    provided = np.array([(i in xi) for i in full_x])
    interpolated_p = 0.15 * np.clip(interpolator_c(full_x), a_min=0, a_max=2*np.max(p)) +\
    0.35 * np.clip(interpolator_l(full_x), a_min=0, a_max=2*np.max(p))  +\
    0.5 * np.clip(interpolator_l(full_x), a_min=0, a_max=2 * np.max(p)) * provided
    return full_x, interpolated_p, ~provided


def one_call_reconstruct(gene_name: List, data_norm: Dict, D: np.ndarray, first_points: List, projs_len: List, mask: np.ndarray, alpha: float=3, beta: float=2) -> np.ndarray:
    # DEPRECATED
    gene_values = [data_norm[name_angle].ix[gene_name].values for name_angle in data_norm.keys()]
    projection_xs = [get_x(data_norm[name_angle]) for name_angle in data_norm.keys()]
    A, b = prepare_regression_symmetry(gene_values, projection_xs, D, first_points, projs_len, verbose=False)
    reconstructor = Reconstructor(alpha=alpha, beta=beta, mask=(mask > 0.2).astype(int))
    return reconstructor.fit_predict(b / b.max(), A)


def IPT_junker(mask: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
               numIter: int = 100) -> np.ndarray:
    """Iterative proportional fitting used by Junker et al."""
    np.seterr(divide='ignore', invalid='ignore')
    
    # projections of 3D mask along sectioning coordinates
    mask_xproj = np.sum(mask, (1, 2))
    mask_yproj = np.sum(mask, (0, 2))
    mask_zproj = np.sum(mask, (0, 1))
    
    # renormalize tomo-seq data by mask volume
    Sx = np.copy(X)  # X * mask_xproj
    Sy = np.copy(Y)  # Y * mask_yproj
    Sz = np.copy(Z)  # Z * mask_zproj
    
    # Basic checks
    Sx[np.isnan(Sx)] = 0
    Sx[np.isinf(Sx)] = np.max(Sx[~np.isinf(Sx)])
    Sy[np.isnan(Sy)] = 0
    Sy[np.isinf(Sy)] = np.max(Sy[~np.isinf(Sy)])
    Sz[np.isnan(Sz)] = 0
    Sz[np.isinf(Sz)] = np.max(Sz[~np.isinf(Sz)])
    
    # normalize total read numbers across 1D data sets
    m = np.mean([np.sum(Sx), np.sum(Sy), np.sum(Sz)])
    Sx = Sx / np.sum(Sx) * m
    Sy = Sy / np.sum(Sy) * m
    Sz = Sz / np.sum(Sz) * m 
    
    # sequential image reconstruction along x, y, z
    R = np.array(mask).astype(float)  # homogenous read density in embryo
    
    for k in range(numIter):
        R_before_update = np.copy(R)
        # redistribute reads along x axis by multiplying with Sx and normalizing
        EyzR = np.sum(R, (1, 2))  # projection along x axis
        R = R * (Sx / EyzR)[:, None, None]
        R[~np.isfinite(R)] = 0  # remove NaNs (division by zero)

        # redistribute reads along x axis by multiplying with Sy and normalizing
        ExzR = np.sum(R, (0, 2))  # projection along y axis
        R = R * (Sy / ExzR)[None, :, None] 
        R[~np.isfinite(R)] = 0  # remove NaNs (division by zero)

        # redistribute reads along x axis by multiplying with Sz and normalizing
        ExyR = np.sum(R, (0, 1))  # projection along x axis
        R = R * (Sz / ExyR)[None, None, :]
        R[~np.isfinite(R)] = 0  # remove NaNs (division by zero)
        
        if np.allclose(R, R_before_update):
            break
    np.seterr(divide='warn', invalid='warn')
    return R


def IPT_2d(mask: np.ndarray, X: np.ndarray, Y: np.ndarray, numIter: int = 100) -> np.ndarray:
    """Iterative proportional fitting used by Junker et al."""
    np.seterr(divide='ignore', invalid='ignore')
    
    # projections of 3D mask along sectioning coordinates
    mask_xproj = np.sum(mask, 1)
    mask_yproj = np.sum(mask, 0)
    
    # renormalize tomo-seq data by mask volume
    Sx = np.copy(X)  # X * mask_xproj
    Sy = np.copy(Y)  # Y * mask_yproj
    
    # Basic checks
    Sx[np.isnan(Sx)] = 0
    Sx[np.isinf(Sx)] = np.max(Sx[~np.isinf(Sx)])
    Sy[np.isnan(Sy)] = 0
    Sy[np.isinf(Sy)] = np.max(Sy[~np.isinf(Sy)])
    
    # normalize total read numbers across 1D data sets
    m = np.mean([np.sum(Sx), np.sum(Sy)])
    Sx = Sx / np.sum(Sx) * m
    Sy = Sy / np.sum(Sy) * m
    
    # sequential image reconstruction along x, y, z
    R = np.array(mask).astype(float)  # homogenous read density in embryo
    
    for k in range(numIter):
        R_before_update = np.copy(R)
        # redistribute reads along x axis by multiplying with Sx and normalizing
        EyR = np.sum(R, 1)  # projection along x axis
        R = R * (Sx / EyR)[:, None]
        R[~np.isfinite(R)] = 0  # remove NaNs (division by zero)

        # redistribute reads along x axis by multiplying with Sy and normalizing
        ExR = np.sum(R, 0)  # projection along y axis
        R = R * (Sy / ExR)[None, :]
        R[~np.isfinite(R)] = 0  # remove NaNs (division by zero)
        
        if np.allclose(R, R_before_update):
            break
    np.seterr(divide='warn', invalid='warn')
    return R


def IPT_angular(angular_projs: List[np.ndarray], D: np.ndarray, proj_len: List[int], sigma: float= None, numIter: int= 100) -> np.ndarray:
    """Iterative proportional fitting adapted for angular cutting"""
    np.seterr(divide='ignore', invalid='ignore')
    cum_proj_len = np.hstack([[0], proj_len]).cumsum()
    D_ = []  # type: list
    for i in range(len(cum_proj_len) - 1):
        D_.append(D[cum_proj_len[i]:cum_proj_len[i + 1], :])

    # renormalize tomo-seq data by mask volume
    S_ = []  # type: list
    for ang in angular_projs:
        S_.append(np.copy(ang))

    # normalize total read numbers across 1D data sets
    m = np.mean([np.sum(S) for S in S_])
    S_ = [S / np.sum(S) * m for S in S_]
    
    # sequential image reconstruction along x, y, z
    R = D_[0].sum(0)  # homogenous read density in embryo
    for k in range(numIter):
        R_before_update = np.copy(R)
        for i in range(len(proj_len)):
            E_R = D_[i].dot(R)
            Rhid = R[None, :] * D_[i] * S_[i][:, None] / E_R[:, None]
            Rhid[~np.isfinite(Rhid)] = 0
            R = Rhid.sum(0)
        
        if np.allclose(R, R_before_update):
            break
        if sigma is None:
            R_img = R.reshape((int(np.sqrt(len(R))), int(np.sqrt(len(R)))))
            R_img = gaussian_filter(R_img, sigma=sigma)
            R = R_img.flat[:]
    np.seterr(divide='warn', invalid='warn')
    return R
