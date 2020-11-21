import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.cluster import KMeans
from gp_extras.kernels import HeteroscedasticKernel
from collections import Counter
from typing import *
from scipy.stats import norm
from .utils import get_plate, get_x


class Normalizer:
    def __init__(self, value_mean: float= 5) -> None:
        self.value_mean = value_mean

    def scaling(self, y: np.ndarray) -> np.ndarray:
        '''Noimport numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.cluster import KMeans
from gp_extras.kernels import HeteroscedasticKernel
from collections import Counter
from typing import *
from scipy.stats import norm
from .utils import get_plate, get_x


class Normalizer:
    def __init__(self, value_mean: float= 5) -> None:
        self.value_mean = value_mean

    def scaling(self, y: np.ndarray) -> np.ndarray:
        '''Normalize/scale y for numerical stability and reusability of the hardcoded parameters of `fit_heteroscedastic`

        Args
        ----
        y: np.ndarray
            the vector to transform

        Returns
        -------
        y_trnasformed: np.ndarray
            y transformed to have average `value_mean`
        '''
        self.factor = y.mean() / self.value_mean
        return y / self.factor

    def inverse_scaling(self, y_transformed: np.ndarray) -> np.ndarray:
        '''Revert the normalization/scaling performed by the `scaling` method

        Args
        ----
         y_trnasformed: np.ndarray
            y transformed to have average `value_mean`

        Returns
        -------
        y: np.ndarray
            the vector originally scaled
        '''
        return self.factor * y_transformed


def fit_heteroscedastic(y: np.ndarray, X: np.ndarray, verbose: bool=True) -> GaussianProcessRegressor:
    """Fit a Gaussian Process with RBF kernel and heteroscedastic noise level
    
    Args
    ----
    y: np.ndarray
        The target variable
    X: np.ndarray
        The independent variables
        
    Returns
    -------
    gp_heteroscedastic: GaussianProcessRegressor
        Model, already fit and ready to predict
        
    Note
    ----
    To get the average function and the predicted noise do:
    y_mean, y_std = gp_heteroscedastic.predict(X_new, return_std=True)
    
    """
    n_clusters = int(np.ceil(len(y) / 10.))
    n_clusters = min(n_clusters, 6)
    km = KMeans(n_clusters=n_clusters)
    labels = km.fit_predict(X)

    prototypes = km.cluster_centers_  # I could explicitelly bin
    # kernel_heteroscedastic = C(1.0, (1e-10, 1000)) * RBF(1, (0.01, 100.0)) + WhiteKernel(1e-3, (1e-10, 50.0))
    kernel_heteroscedastic = C(5.0, (5e-1, 5e2)) * RBF(7, (4, len(y) / 1.5)) + HeteroscedasticKernel.construct(prototypes, 0.5, (1e-6, 10.0), gamma=2., gamma_bounds="fixed")
    gp_heteroscedastic = GaussianProcessRegressor(kernel=kernel_heteroscedastic, alpha=0, n_restarts_optimizer=5)
    gp_heteroscedastic.fit(X, y)
    if verbose:
        print("n_clusters: %d, min_size: %d" % (n_clusters, min(Counter(labels).values())))
        print("Heteroscedastic kernel: %s" % gp_heteroscedastic.kernel_)
        print("Heteroscedastic LML: %.3f" % gp_heteroscedastic.log_marginal_likelihood(gp_heteroscedastic.kernel_.theta))
    
    return gp_heteroscedastic


def normalization_factors(mu1: float, mu2: float, std1: float, std2: float, q: float=0.15) -> Tuple[float, float]:
    ''' Given the average value of a gaussian process at its extremities it returns
    the normalization factors that the two sides needs to be multiplied to so that
    the probablity of the smaller extremity generating a realization bigger of the one of the bigger extremity
    is at least q (default 15%)

    Args
    ----
    mu1: float
        estimate average of the first extremity point
    mu2: float
        estimate average of the second extremity point
    std1: float
        extimation of the standard deviation around the first extremity point
    std2: float
        extimation of the standard deviation around the second extremity point
    
    Returns
    -------
    factor1: float
        number that should be mutliplied by the first extremity side of the function
    factor2: float
        number that should be mutliplied by the second extremity side of the function
    '''
    ixs = np.argsort([mu1, mu2])
    mu_smaller, mu_bigger = np.array([mu1, mu2])[ixs]
    scale_of_smaller, scale_of_bigger = np.array([std1, std2])[ixs]
    shift = -min(0, norm.isf(q=q, loc=mu_smaller - mu_bigger, scale=np.sqrt(scale_of_smaller**2 + scale_of_bigger**2)))
    mu_smaller_updated = mu_smaller + shift / 2.
    mu_bigger_updated = mu_bigger - shift / 2.
    factor_smaller, factor_bigger = mu_smaller_updated / mu_smaller, mu_bigger_updated / mu_bigger
    factor1, factor2 = np.array([factor_smaller, factor_bigger])[ixs]
    return factor1, factor2


def predict_gp_heteroscedastic(y: np.ndarray, X: np.ndarray, X_new: np.ndarray, verbose: bool=True) -> Tuple[np.ndarray, np.ndarray]:
    """Fits and predict an heteroscedastic-kernel gaussian process model, taking care of the normalizations

    Args
    ----
    y: np.ndarray
        the measurements y to fit by the gaussian process
    X: np.ndarray
        The points at whic the measurments y are available
    X_new: np.ndarray
        The points where to predict from the model
    verbose: bool default True
        if you want to print info on the fit results

    Returns
    -------
    y_mean: np.ndarray
        The mean funtion of the gp at each point X_new
    y_std: np.ndarray
        The deviation from the mean at every point X_new
    """
    nrml = Normalizer(value_mean=5)
    y_norm = nrml.scaling(y)
    model = fit_heteroscedastic(y_norm, X, verbose)
    y_mean, y_std = model.predict(X_new, return_std=True)
    y_mean = nrml.inverse_scaling(y_mean)
    y_std = nrml.inverse_scaling(y_std)
    return y_mean, y_std


def use_gp_to_adjust_plates_diff(data: Dict[str, pd.DataFrame], q: float=0.15) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """ Fits a GP to the left and right plate and corrects for different depth.
    It does this by matching the extremities in a conservative way uses a statistical condition.

    Args
    ----
    data: Dict[str, pd.DataFrame]
        Dictionary on anfles datasets
    q : float
        Value used to decide how much to adjust the plate difference
        The probability that a lower margin rv is bigger than the higher margin rv
        0 -> no adjustment
        0.5 -> exact matching of mean functions

    Returns
    -------
    coordinates_dict: Tuple[Dict[str, Any], Dict[str, Any]]
        To pass to the plotting fucntion.
        It contains the values x, y, X_news, y_means, y_stds, f
        Where each of them is a list of two elements containin the arrays corresponding to each of the plates

    normalizations: Dict[str, Tuple[float, float]]
        keys are angles and values are tuple containing the normalizations factors that should be multiplied with that two plates respectivelly

    """
    normalizations = {}
    coordinates_dict = {}
    for i, name_angle in enumerate(data.keys()):
        plate = get_plate(data[name_angle])
        n_plates = len(set(plate))
        X_news = []
        y_means = []
        y_stds = []
        xall = get_x(data[name_angle])
        yall = data[name_angle].sum(0)
        x = [xall[plate == plate_i + 1] for plate_i in range(n_plates)]
        y = [yall[plate == plate_i + 1] for plate_i in range(n_plates)]
        
        # For each plate
        for j in range(n_plates):
            # Fit Etheroscedastic model
            X_new = np.linspace(min(x[j]), max(x[j]), 80)[:, None]
            y_new, y_std = predict_gp_heteroscedastic(y[j], x[j][:, None], X_new, verbose=False)
            # Save the extremity values
            X_news.append(X_new)
            y_means.append(y_new)
            y_stds.append(y_std)
        
        # Compute the normalization factor and store it
        temp_fs = []
        for junct_i in range(n_plates - 1):
            ext_mus = y_means[junct_i][-1], y_means[junct_i + 1][1]
            ext_stds = y_stds[junct_i][-1], y_stds[junct_i + 1][1]
            f = normalization_factors(ext_mus[0], ext_mus[1], ext_stds[0], ext_stds[1], q=q)
            temp_fs.append(list(f))

        N = len(temp_fs)
        for k in range(N - 1):
            temp_fs[k + 1][1] = temp_fs[k + 1][1] * temp_fs[k][-1] / temp_fs[k + 1][0]
            temp_fs[k + 1][0] = temp_fs[k][-1]

        fs = []
        for k in range(N):
            fs.append(temp_fs[k][0])
        fs.append(temp_fs[-1][-1])
        
        normalizations[name_angle] = fs
        coordinates_dict[name_angle] = x, y, X_news, y_means, y_stds, fs

    return coordinates_dict, normalizations


def use_gp_to_adjust_spikes_diff(spikes: Dict[str, pd.DataFrame], q: float=0.15) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """ Fits a GP to the left and right plate spikes and corrects to the avareage

    Args
    ----
    spikes: Dict[str, pd.DataFrame]
        Dictionary on anfles datasets
    q : float
        Value used to decide how much to adjust the plate difference
        The probability that a lower margin rv is bigger than the higher margin rv
        0 -> no adjustment
        0.5 -> exact matching of mean functions

    Returns
    -------
    coordinates_dict: Tuple[Dict[str, Any], Dict[str, Any]]
        To pass to the plotting fucntion.
        It contains the values x, y, X_news, y_means, y_stds, adj
        Where each of them is a list of two elements containin the arrays corresponding to each of the plates

    adjustments: Dict[str, Tuple[float, float]]
        the number that needs to be multiplied by each point to obtain the normalization

    """
    adjustments = {}
    coordinates_dict = {}
    for i, name_angle in enumerate(spikes.keys()):
        plate = get_plate(spikes[name_angle])
        n_plates = len(set(plate))
        X_news = []
        y_pred = []
        y_stds = []
        xall = get_x(spikes[name_angle])
        yall = spikes[name_angle].sum(0)
        x = [xall[plate == plate_i + 1] for plate_i in range(n_plates)]
        y = [yall[plate == plate_i + 1] for plate_i in range(n_plates)]
        
        # For each plate
        for j in range(n_plates):
            # Fit Etheroscedastic model
            X_new = x[j][:, None]  # np.linspace(min(x[j]), max(x[j]), 80)[:, None]
            y_new, y_std = predict_gp_heteroscedastic(y[j], x[j][:, None], X_new, verbose=False)
            # Save the extremity values
            X_news.append(X_new)
            y_pred.append(y_new)
            y_stds.append(y_std)

        mu = np.mean(np.concatenate([y_pred[x] for x in range(n_plates)], axis=0))
        delta_y = [y_pred[x] - mu for x in range(n_plates)]
        adj = [1 - (delta_y[x] / y_pred[x]) for x in range(n_plates)]
        adjustments[name_angle] = np.concatenate(adj)
        coordinates_dict[name_angle] = x, y, X_news, y_pred, y_stds, adj

    return coordinates_dict, adjustments
rmalize/scale y for numerical stability and reusability of the hardcoded parameters of `fit_heteroscedastic`

        Args
        ----
        y: np.ndarray
            the vector to transform

        Returns
        -------
        y_trnasformed: np.ndarray
            y transformed to have average `value_mean`
        '''
        self.factor = y.mean() / self.value_mean
        return y / self.factor

    def inverse_scaling(self, y_transformed: np.ndarray) -> np.ndarray:
        '''Revert the normalization/scaling performed by the `scaling` method

        Args
        ----
         y_trnasformed: np.ndarray
            y transformed to have average `value_mean`

        Returns
        -------
        y: np.ndarray
            the vector originally scaled
        '''
        return self.factor * y_transformed


def fit_heteroscedastic(y: np.ndarray, X: np.ndarray, verbose: bool=True) -> GaussianProcessRegressor:
    """Fit a Gaussian Process with RBF kernel and heteroscedastic noise level
    
    Args
    ----
    y: np.ndarray
        The target variable
    X: np.ndarray
        The independent variables
        
    Returns
    -------
    gp_heteroscedastic: GaussianProcessRegressor
        Model, already fit and ready to predict
        
    Note
    ----
    To get the average function and the predicted noise do:
    y_mean, y_std = gp_heteroscedastic.predict(X_new, return_std=True)
    
    """
    n_clusters = int(np.ceil(len(y) / 10.))
    n_clusters = min(n_clusters, 6)
    km = KMeans(n_clusters=n_clusters)
    labels = km.fit_predict(X)

    prototypes = km.cluster_centers_  # I could explicitelly bin
    # kernel_heteroscedastic = C(1.0, (1e-10, 1000)) * RBF(1, (0.01, 100.0)) + WhiteKernel(1e-3, (1e-10, 50.0))
    kernel_heteroscedastic = C(5.0, (5e-1, 5e2)) * RBF(7, (4, len(y) / 1.5)) + HeteroscedasticKernel.construct(prototypes, 0.5, (1e-6, 10.0), gamma=2., gamma_bounds="fixed")
    gp_heteroscedastic = GaussianProcessRegressor(kernel=kernel_heteroscedastic, alpha=0, n_restarts_optimizer=5)
    gp_heteroscedastic.fit(X, y)
    if verbose:
        print("n_clusters: %d, min_size: %d" % (n_clusters, min(Counter(labels).values())))
        print("Heteroscedastic kernel: %s" % gp_heteroscedastic.kernel_)
        print("Heteroscedastic LML: %.3f" % gp_heteroscedastic.log_marginal_likelihood(gp_heteroscedastic.kernel_.theta))
    
    return gp_heteroscedastic


def normalization_factors(mu1: float, mu2: float, std1: float, std2: float, q: float=0.15) -> Tuple[float, float]:
    ''' Given the average value of a gaussian process at its extremities it returns
    the normalization factors that the two sides needs to be multiplied to so that
    the probablity of the smaller extremity generating a realization bigger of the one of the bigger extremity
    is at least q (default 15%)

    Args
    ----
    mu1: float
        estimate average of the first extremity point
    mu2: float
        estimate average of the second extremity point
    std1: float
        extimation of the standard deviation around the first extremity point
    std2: float
        extimation of the standard deviation around the second extremity point
    
    Returns
    -------
    factor1: float
        number that should be mutliplied by the first extremity side of the function
    factor2: float
        number that should be mutliplied by the second extremity side of the function
    '''
    ixs = np.argsort([mu1, mu2])
    mu_smaller, mu_bigger = np.array([mu1, mu2])[ixs]
    scale_of_smaller, scale_of_bigger = np.array([std1, std2])[ixs]
    shift = -min(0, norm.isf(q=q, loc=mu_smaller - mu_bigger, scale=np.sqrt(scale_of_smaller**2 + scale_of_bigger**2)))
    mu_smaller_updated = mu_smaller + shift / 2.
    mu_bigger_updated = mu_bigger - shift / 2.
    factor_smaller, factor_bigger = mu_smaller_updated / mu_smaller, mu_bigger_updated / mu_bigger
    factor1, factor2 = np.array([factor_smaller, factor_bigger])[ixs]
    return factor1, factor2


def predict_gp_heteroscedastic(y: np.ndarray, X: np.ndarray, X_new: np.ndarray, verbose: bool=True) -> Tuple[np.ndarray, np.ndarray]:
    """Fits and predict an heteroscedastic-kernel gaussian process model, taking care of the normalizations

    Args
    ----
    y: np.ndarray
        the measurements y to fit by the gaussian process
    X: np.ndarray
        The points at whic the measurments y are available
    X_new: np.ndarray
        The points where to predict from the model
    verbose: bool default True
        if you want to print info on the fit results

    Returns
    -------
    y_mean: np.ndarray
        The mean funtion of the gp at each point X_new
    y_std: np.ndarray
        The deviation from the mean at every point X_new
    """
    nrml = Normalizer(value_mean=5)
    y_norm = nrml.scaling(y)
    model = fit_heteroscedastic(y_norm, X, verbose)
    y_mean, y_std = model.predict(X_new, return_std=True)
    y_mean = nrml.inverse_scaling(y_mean)
    y_std = nrml.inverse_scaling(y_std)
    return y_mean, y_std


def use_gp_to_adjust_plates_diff(data: Dict[str, pd.DataFrame], q: float=0.15) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """ Fits a GP to the left and right plate and corrects for different depth.
    It does this by matching the extremities in a conservative way uses a statistical condition.

    Args
    ----
    data: Dict[str, pd.DataFrame]
        Dictionary on anfles datasets
    q : float
        Value used to decide how much to adjust the plate difference
        The probability that a lower margin rv is bigger than the higher margin rv
        0 -> no adjustment
        0.5 -> exact matching of mean functions

    Returns
    -------
    coordinates_dict: Tuple[Dict[str, Any], Dict[str, Any]]
        To pass to the plotting fucntion.
        It contains the values x, y, X_news, y_means, y_stds, f
        Where each of them is a list of two elements containin the arrays corresponding to each of the plates

    normalizations: Dict[str, Tuple[float, float]]
        keys are angles and values are tuple containing the normalizations factors that should be multiplied with that two plates respectivelly

    """
    normalizations = {}
    coordinates_dict = {}
    for i, name_angle in enumerate(data.keys()):
        plate = get_plate(data[name_angle])
        n_plates = len(set(plate))
        X_news = []
        y_means = []
        y_stds = []
        xall = get_x(data[name_angle])
        yall = data[name_angle].sum(0)
        x = [xall[plate == plate_i + 1] for plate_i in range(n_plates)]
        y = [yall[plate == plate_i + 1] for plate_i in range(n_plates)]
        
        # For each plate
        for j in range(n_plates):
            # Fit Etheroscedastic model
            X_new = np.linspace(min(x[j]), max(x[j]), 80)[:, None]
            y_new, y_std = predict_gp_heteroscedastic(y[j], x[j][:, None], X_new, verbose=False)
            # Save the extremity values
            X_news.append(X_new)
            y_means.append(y_new)
            y_stds.append(y_std)
        
        # Compute the normalization factor and store it
        temp_fs = []
        for junct_i in range(n_plates - 1):
            ext_mus = y_means[junct_i][-1], y_means[junct_i + 1][1]
            ext_stds = y_stds[junct_i][-1], y_stds[junct_i + 1][1]
            f = normalization_factors(ext_mus[0], ext_mus[1], ext_stds[0], ext_stds[1], q=q)
            temp_fs.append(list(f))

        N = len(temp_fs)
        for k in range(N - 1):
            temp_fs[k + 1][1] = temp_fs[k + 1][1] * temp_fs[k][-1] / temp_fs[k + 1][0]
            temp_fs[k + 1][0] = temp_fs[k][-1]

        fs = []
        for k in range(N):
            fs.append(temp_fs[k][0])
        fs.append(temp_fs[-1][-1])
        
        normalizations[name_angle] = fs
        coordinates_dict[name_angle] = x, y, X_news, y_means, y_stds, fs

    return coordinates_dict, normalizations


def use_gp_to_adjust_spikes_diff(spikes: Dict[str, pd.DataFrame], q: float=0.15) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """ Fits a GP to the left and right plate spikes and corrects to the avareage

    Args
    ----
    spikes: Dict[str, pd.DataFrame]
        Dictionary on anfles datasets
    q : float
        Value used to decide how much to adjust the plate difference
        The probability that a lower margin rv is bigger than the higher margin rv
        0 -> no adjustment
        0.5 -> exact matching of mean functions

    Returns
    -------
    coordinates_dict: Tuple[Dict[str, Any], Dict[str, Any]]
        To pass to the plotting fucntion.
        It contains the values x, y, X_news, y_means, y_stds, adj
        Where each of them is a list of two elements containin the arrays corresponding to each of the plates

    adjustments: Dict[str, Tuple[float, float]]
        the number that needs to be multiplied by each point to obtain the normalization

    """
    adjustments = {}
    coordinates_dict = {}
    for i, name_angle in enumerate(spikes.keys()):
        plate = get_plate(spikes[name_angle])
        n_plates = len(set(plate))
        X_news = []
        y_pred = []
        y_stds = []
        xall = get_x(spikes[name_angle])
        yall = spikes[name_angle].sum(0)
        x = [xall[plate == plate_i + 1] for plate_i in range(n_plates)]
        y = [yall[plate == plate_i + 1] for plate_i in range(n_plates)]
        
        # For each plate
        for j in range(n_plates):
            # Fit Etheroscedastic model
            X_new = x[j][:, None]  # np.linspace(min(x[j]), max(x[j]), 80)[:, None]
            y_new, y_std = predict_gp_heteroscedastic(y[j], x[j][:, None], X_new, verbose=False)
            # Save the extremity values
            X_news.append(X_new)
            y_pred.append(y_new)
            y_stds.append(y_std)

        mu = np.mean(np.concatenate([y_pred[x] for x in range(n_plates)], axis=0))
        delta_y = [y_pred[x] - mu for x in range(n_plates)]
        adj = [1 - (delta_y[x] / y_pred[x]) for x in range(n_plates)]
        adjustments[name_angle] = np.concatenate(adj)
        coordinates_dict[name_angle] = x, y, X_news, y_pred, y_stds, adj

    return coordinates_dict, adjustments
