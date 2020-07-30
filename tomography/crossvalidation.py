from typing import *
import numpy as np
from itertools import combinations, product, permutations
from scipy.special import loggamma


def rss_objective( y_test: np.ndarray, y_predicted: np.ndarray) -> float:
    """Return the residual sum of squares"""
    return np.sum((y_predicted - y_test)**2)


def rmse_objective( y_test: np.ndarray, y_predicted: np.ndarray) -> float:
    """Return the residual sum of squares"""
    return np.mean((y_predicted - y_test)**2)

def corr_objective( y_test: np.ndarray, y_predicted: np.ndarray) -> float:
    """Return the negative correlation (to minimize)"""
    return -np.corrcoef(y_predicted.flat[:], y_test.flat[:])[0,1]

def new_obj(y_test, y_predicted, f):
    print ("here")
    return f*corr_objective(y_test, y_predicted) #+ rmse_objective(y_test, y_predicted)

def poisson_log_lik_objective(y_test: np.ndarray, y_predicted: np.ndarray) -> float:
    """Minus Log likelihood 
    $l(\lambda;x)=\sum\limits^n_{i=1}x_i \text{ log }\lambda-n\lambda$
    """
    return -np.sum(y_test * np.log(y_predicted) - y_predicted)


def llf(y, mu, r):
    a1 = mu/r
    a2 = mu + a1
    llf = (loggamma(y + a1) - loggamma(y + 1) - loggamma(a1) + a1 * np.log(a1) + y * np.log(mu) - (y + a1) * np.log(a2) )
    return -np.sum(llf)


def nb_loglik3(y, mu, psi):

    psi_min = 1. / psi
    lggamma_fun_ratio = loggamma(y + psi_min) - loggamma(psi_min) - loggamma(y + 1)
    log1mupsi = np.log(1 + mu * psi)
    lgf1 = - psi_min * log1mupsi
    lgf2 = y * (np.log(mu) - log1mupsi + np.log(psi))

    return -np.sum(lggamma_fun_ratio + lgf1 + lgf2)

def nb_loglik(y, mu, r):
    """
   Continuous Negative binomial loglikelihood function. Numerically stable implementation.

   Arguments
   ---------
   y: float or np.ndarray
       The values to evaluate the loglikehood on
   mu: float or np.ndarray
       The mean parameter of the negative binomial distribution, is y_predicted
   psi: float or np.ndarray
       The psi parameter of the NB distribution
       It corresponds to (VAR[x] - E[x])/ E[x]**2
       For a constant overdispersion `r` set it to r/mu

   Returns
   -------
   The Negative binomial LogLikelihood

   Note
   ----
   For more information on Continuous negative binomial likelihood function:
   - Robinson and Smyth, Biostatistics 2007
   Stability to high/low input values has been tested manually but there are no theoretical guarantees

   """
   #print (mu)

    psi = r/(mu)
    psi_min = 1. / psi
    lggamma_fun_ratio = loggamma(y + psi_min) - loggamma(psi_min) - loggamma(y + 1)
    log1mupsi = np.log(1 + mu * psi)
    lgf1 = - psi_min * log1mupsi
    lgf2 = y * (np.log(mu) - log1mupsi + np.log(psi))

    return -np.sum(lggamma_fun_ratio + lgf1 + lgf2)

def split_list(lista: List, split_size: Tuple[int, int]) -> Iterator[Sequence[Any]]:
    """Split a list in two groups of defined size in all possible permutations of combinations

    Args
    ----
    lista: list
        list ot be split

    split_size: Tuple[int, int]
        a tuple of two integers , their sum neews to be len(lista)

    Return
    ------
    combinations: Itarator[Tuple[List, List]]
        iterators of the possible splits for example ((1,2,3), (4,5)), ((1,2,4), (3,5)), ...
    """
    for i in combinations(lista, split_size[0]):
        left = tuple(set(lista) - set(i))
        for j in combinations(left, split_size[1]):
            yield i, j


def bool_from_interval(intervals_ixs: List[int], boundaries: np.ndarray, simmetry: bool=True) -> np.ndarray:
    '''Given interval to include and boundaries returns an array that can be used for bool indexing.

    Args
    ----
    intervals_ixs: list
        A list of integers of which interval include, for example if intervals_ixs = [0,3] you want to include only 
        data with ix so that boundaries[0] <= ix < boundaries[1] & boundaries[3] <= ix < boundaries[4]
    
    boundaries: np.ndarray
        an array indicating the borders of the boundaries
    
    simmetry: bool
        if True will adapt the result to the simmetery constrained problem

    Returns
    -------
    bool_filter: np.ndarray of bool
        a boolean array that can be used for filtering
    '''
    inboundary_i = np.digitize(np.arange(max(boundaries)), boundaries) - 1
    bool_filter = np.in1d(inboundary_i, intervals_ixs)
    if simmetry:
        bool_filter = np.hstack([bool_filter, bool_filter])
    return bool_filter


def cross_validate(A: np.ndarray, b: np.ndarray, mask: np.ndarray, boundaries: np.ndarray, alpha_beta_grid: List[List[float]],
                   score_f: Callable, reconstructor_class: Callable) -> List[List[float]]:
    """Slow but exhaustive crossvalidation by naive grid search and no optimization warmstart

    Args
    ----
    A: np.ndarray
        the design matrix (as returned by a variant of tomography.prepare_regression function)
    b: np.ndarray
        the observation vector (as returned by a variant of tomography.prepare_regression function)
    mask: np.ndarray
        grayscale mask
    boundaries: np.ndarray
        array constaining the borders of the intervals of b corresponding to different projections (starting from 0)
    alpha_beta_grid : List[List[float]]
        a list of list containing the alpha, beta values to try
    score_f: Callable
        function taking two arguments (b_test, b_train) and returing the score to be calulated
    reconstructor_class: class default(ReconstructorFast)
        should be either Reconstructor or ReconstructorFast Note: class not instance

    Returns
    -------
    all_scores: List[List[float]]
        the result of calling score_f for every possible split for every element of the grid
    
    """
    b1 = b / b.max()  # do this normalization in case b was not already normalized
    # it makes sure we are working with the same scale for all the splits
    all_scores = []  # typle: List
    for alpha, beta in alpha_beta_grid:
        scores = []
        print("alpha: %s beta: %s" % (alpha, beta))
        for (train_list, test_list) in split_list(list(range(5)), (4, 1)):
            trainset_bool = bool_from_interval(train_list, boundaries)
            testset_bool = bool_from_interval(test_list, boundaries)
            A_train, b_train = A[trainset_bool, :], b1[trainset_bool]
            A_test, b_test = A[testset_bool, :], b1[testset_bool]
            reconstructor = reconstructor_class(alpha=alpha, beta=beta, mask=(mask > 0.2).astype(int))
            result = np.array(reconstructor.fit(b_train, A_train).x.value).flat[:]
            scores.append(score_f(b_test, A_test.dot(result)))
        all_scores.append(scores)
    return all_scores