from __future__ import division, print_function

from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
from scipy.signal import correlate
from IPython.display import display
import tomography
from tomography.core import build_Design_Matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from backspinpy import CEF_obj
from sklearn.svm import SVR
from scipy import stats
import scipy.sparse
import pickle
from scipy.integrate import simps
from skimage.measure import find_contours
from skimage import morphology
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ShuffleSplit
from tomography import get_x, get_plate, colorize, normalize_AUC

import h5py

def autocorr_shift(source, target, shift_lim=None):
    if shift_lim is None:
        return np.argmax(correlate(target, source)) - (len(source)-1)
    else:
        autoR = correlate(target, source)
        indexes = np.argsort(autoR)[::-1] - (len(source)-1)
        for i in indexes:
            if i >= shift_lim[0] and i <= shift_lim[1]:
                return i

def equilibrate(a, b):
    le = max(len(a), len(b))
    tempa, tempb = np.zeros((le,)), np.zeros((le,)) 
    tempa[:len(a)] = a
    tempb[:len(b)] = b
    return tempa, tempb

def interpolate(xi, p):
    full_x = np.arange(xi[0], xi[-1] + 1)
    linear_interpolator = interp1d(xi, p, kind='linear', fill_value=0, bounds_error=False)
    return full_x, linear_interpolator(full_x)

def gauss_kernel(x, mu=0, sigma=1):
    return np.exp(- 0.5 * (x - mu)**2 / sigma**2)
    #return scipy.stats.norm.pdf(x, loc=mu, scale=sigma)

def bilateral_filter(y, sigma_s=2.5, sigma_r_fold=2.5):
    x = np.arange(len(y))
    fr = gauss_kernel(y, mu=y[:,None], sigma=sigma_r_fold * np.mean(np.abs(np.diff(y[y>0]))))
    gs = gauss_kernel(x, mu=x[:,None], sigma=sigma_s)
    return (y[:,None] * fr * gs).sum(0) / (fr * gs).sum(0)

def bilateral_kernel_show(y, sigma_s=2.5, sigma_r_fold=2.5):
    x = np.arange(len(y))
    fr = gauss_kernel(y, mu=y[:,None], sigma=sigma_r_fold * np.mean(np.abs(np.diff(y[y>0]))))
    gs = gauss_kernel(x, mu=x[:,None], sigma=sigma_s)
    plt.imshow(fr*gs, interpolation="bilinear")

def score_prediction(y, y_pred, shift_lim=None):
    # DATA
    y = np.pad(y, (6,6), mode="constant")
    y = bilateral_filter(y)
    y = y / np.mean(y[y>np.percentile(y[y>0], 80)])

    # PREDICTED
    y_pred = np.pad(y_pred, (6,6), mode="constant")
    y_pred = bilateral_filter(y_pred)
    y_pred = y_pred / np.mean(y_pred[y_pred>np.percentile(y_pred[y_pred>0], 80)])

    # DETERMINE SHIFT
    x_shift = autocorr_shift(y, y_pred, shift_lim=shift_lim)
    
    # APPLY SHIFT
    if x_shift < 0:
        if len(y_pred) < len(y):
            y = y[-x_shift:-x_shift+len(y_pred)]
            y_pred = y_pred[:len(y)]
        elif len(y_pred) >= len(y):
            y = y[-x_shift:]
            y_pred = y_pred[:len(y)]
    elif x_shift >= 0:
        if len(y_pred) < len(y):
            y = np.pad(y, (x_shift, 0), mode="constant")[:len(y_pred)]
            y_pred = y_pred
        elif len(y_pred) >= len(y):
            y = np.pad(y, (x_shift, 0), mode="constant")[:len(y_pred)]
            y_pred = y_pred[:len(y)]

    # CALCULATE SCORES
    corr_score = 1-np.corrcoef(y[~((y==0) & (y_pred==0))], y_pred[~((y==0) & (y_pred==0))])[0,1]
    L2_score = np.sum((y - y_pred)**2)
    return corr_score, L2_score, y, y_pred, x_shift

def grid_score_prediction(angles_probes, wid_probes, D_cache, ref_im_flat, y, pw, return_ys=False):
    record = {"corr_score": np.zeros((len(angles_probes), len(wid_probes))),
              "L2_score": np.zeros((len(angles_probes), len(wid_probes))),
              "y": [[] for i in range(len(angles_probes))],
              "y_pred": [[] for i in range(len(angles_probes))],
              "x_shift": np.zeros((len(angles_probes), len(wid_probes)))}
    for i, angle in enumerate(angles_probes):
        for j, wid in enumerate(wid_probes):
            predicted_angle = angle
            predicted_wid = wid
            y_pred = D_cache[(predicted_angle, predicted_wid)] @ ref_im_flat

            corr_score, L2_score, y_hat, y_pred_hat, x_shift = score_prediction(y, y_pred,
                                                                                shift_lim=pw)
            record["corr_score"][i, j] = corr_score
            record["L2_score"][i, j]  = L2_score
            if return_ys:
                record["y"][i].append(y_hat)
                record["y_pred"][i].append(y_pred_hat)
            record["x_shift"][i, j]  = x_shift
    return record

class DCache(object):
    def __init__(self, mask):
        self.D = {}
        self.mask = mask
        
    def __getitem__(self, key):
        try:
            return self.D[key]
        except KeyError:
            rad, wid = np.deg2rad(key[0]), key[1]
            self.D[key] = build_Design_Matrix(np.array([rad]), [wid], 1. * self.mask, return_sparse=True)[0]
            return self.D[key]