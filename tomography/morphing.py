import numpy as np
from scipy.misc import comb
from scipy.spatial import KDTree
from numpy.linalg import norm
from skimage import measure
from scipy.optimize import minimize
from typing import *


def make_bounding_arches(points: np.ndarray, mask: np.ndarray, epsilon: float, return_filtered_bool: bool=True) -> Any:
    """Split the countour of the mask in bounding arches given some points
    
    Args
    ----
    points : np.ndarray shape=(N,2)
        the points on the mask
    mask : np.ndarray
        a binary or float mask, the contour at 0.5 will be interpreted as border
    epsilon: float
        distance in pixels
    return_filtered_bool: bool
        wether to return filtered_bool
        
    Return
    ------
    arches: np.ndarray
        The contour of the mask divided in arches 
        Arches are chosen so that they start from the nearest neighbour of the closest point
        only points within epsilon from the contour are considered when forming arches
    filtered_bool: np.ndarray
        boolean of the points that have been selected as borders of the archers
    """
    contours = measure.find_contours(mask, 0.5)
    assert len(contours) == 1, "Two or more contours were obtained, check your mask"
    contour = contours[0]
    kd = KDTree(contour)
    dd, ix = kd.query(points, k=1)
    filtered = dd < epsilon
    assert np.sum(filtered) >= 2, "The epsilon chosen is too small, less than two points have been found on the contour"
    ix = ix[filtered]
    ix = np.unique(ix)
    ix.sort()
    arches = []
    for i in range(len(ix) - 1):
        arches.append(contour[ix[i]:ix[i + 1], :])
    arches.append(np.vstack([contour[ix[-1]:, :], contour[:ix[0], :]]))
    if return_filtered_bool:
        return filtered, arches
    else:
        return arches


def bernstein_poly(i: int, n: int, t: np.ndarray) -> np.ndarray:
    """ The Bernstein polynomial of n, i as a function of t.  B_ni(t) """
    return comb(n, i) * (t**(n - i)) * (1 - t)**i


def bezier_curv_eval(control_points: np.ndarray, k: int=100) -> np.ndarray:
    """Control points
    
    Args
    ----
    control_points: np.ndarray shape=(n_points, dimensions)
        the control points that define a bezier curve
        
    k: int
        number of intermediate points at wich it is evaluated
    
    Returns
    -------
    curve: np.ndarray
    """
    n = control_points.shape[0]
    t = np.linspace(0, 1, k)
    polynomials_M = bernstein_poly(np.arange(n)[:, None], n-1, t)
    return np.dot(polynomials_M.T, control_points)


class ControlPointsFitter:
    def __init__(self, n_ctrl_points: int=3, dim: int=2, kind: str="paired", resolution: int=None) -> None:
        """ Fits internal control points of a bezier curve
        
        Args
        ----
        
        n_points: int
            the number of control points
            
        dim: int
            the dimensionality of the space
            
        kind: str
            kind of optimization to be used can be paired or nearest_neighbour
            `paired` assumes compares the ith arc point to the ith point in the bezier curve
            `nearest_neighbour` matches the arc points to the nearest bezier curve points
            
        resolution: int (used only when kind="nearest_neighbour")
            number of bezier points to evaluate, if None use the number of points in the arch
        
        """
        self.n_ctrl_points = n_ctrl_points
        self.dim = dim
        self.control_points = np.zeros((n_ctrl_points, dim))
        self.kind = kind
        self.resolution = resolution
        
    def fit(self, arc: np.ndarray) -> np.ndarray:
        """ Fits self.n_points-2 control points fiven an arc
        
        Args
        ----
        arc: np.ndarray shape=(n_arc_points, self.dim)
            The arc that will be fit
        
        Returns
        -------
        ctrl_points: np.ndarray shape=(self.n_ctrl_points, self.dim)
        
        """
        
        self.arc = arc
        if self.resolution is None:
            self.resolution = self.arc.shape[0]
        self.mN = arc.shape[0]//2
        if self.kind == "nearest_neighbour":
            self.t = np.linspace(0, 1, self.resolution)
            # Prepare a kd tree for neighreast neighbours
            self.kdt = KDTree(self.arc)
        else:
            self.t = np.linspace(0, 1, arc.shape[0])
        
        if self.n_ctrl_points == 3:
            # The point in the middle
            self.x0 = [arc[self.mN, 0], arc[self.mN,1]]
        elif self.n_ctrl_points == 4:
            # The average between the point in the middle and the extremities
            self.x0 = [0.5 * (arc[self.mN, 0] + arc[0, 0]), 0.5 * (arc[self.mN, 1] + arc[0, 1]),
                       0.5 * (arc[self.mN, 0] + arc[-1, 0]), 0.5 * (arc[self.mN, 1] + arc[-1, 1])]
        else:
            # Ordered internal points
            self.x0 = [(1. / (self.n_ctrl_points - 2)) * (i * arc[arc.shape[0] // (self.n_ctrl_points - 2), j]) for i in range(1, self.n_ctrl_points - 2 + 1)
                       for j in range(2)]
        
        # Prepare the bernstein polinomial matrix
        self.polynomials_M = bernstein_poly(np.arange(self.n_ctrl_points)[:, None], self.n_ctrl_points - 1, self.t)
        
        self.control_points[0, :] = self.arc[0, :]
        self.control_points[-1, :] = self.arc[-1, :]
        self.optimization = minimize(self._objective, self.x0)  # args=(self.arc, self.kdt, self.polynomials_M )
        
    def _objective(self, x: Tuple) -> float:
        control_points = np.copy(self.control_points)
        control_points[1:-1, :] = np.reshape(x, (len(x) // 2, 2))
        new_points = self.polynomials_M.T.dot(control_points)
        if self.kind == "paired":
            RSS = norm(new_points - self.arc, ord="fro")
        elif self.kind == "nearest_neighbour":
            RSS = np.sqrt(np.sum((self.kdt.query(new_points)[0])**2))  # I take the square only to be comparable to nearest_neighbour
        return RSS
        
    def fit_predict(self, arc: np.ndarray, only_new: bool=False) -> np.ndarray:
        """ Fits self.n_points-2 control points fiven an arc
        
        Args
        ----
        arc: np.ndarray shape=(n_arc_points, self.dim)
            The arc that will be fit
        only_new: bool
            weather to return only the fitted internal points or all the control points
        
        Returns
        -------
        ctrl_points: np.ndarray shape=(self.n_ctrl_points, self.dim)
        
        """
        self.fit(arc)
        if only_new:
            return np.reshape(self.optimization.x, (len(self.optimization.x) // 2, 2))
        else:
            self.control_points[1:-1, :] = np.reshape(self.optimization.x, (len(self.optimization.x) // 2, 2))
            self.control_points[0, :] = self.arc[0, :]
            self.control_points[-1, :] = self.arc[-1, :]
            return self.control_points
        
    def new_bezier_points(self) -> np.ndarray:
        return self.polynomials_M.T.dot(self.control_points)


def generate_new_points(points: np.ndarray, mask: np.ndarray, epsilon: float, fitter_kind: str='nearest_neighbour', extra_per_arch: int=1,
                        resolution: int=None, return_filtered_bool: bool=True) -> Any:
    """Add new reference point outside the mask ensuring a good delaunay triangulation
    It uses the internal control points of a bezier curve

    Args
    ----
    points : np.ndarray shape=(N,2)
    the points on the mask

    mask : np.ndarray
        a binary or float mask, the contour at 0.5 will be interpreted as border

    epsilon: float
        distance in pixels

    kind: str
        kind of optimization to be used can be paired or nearest_neighbour
        `paired` compares the ith arc point to the ith point in the bezier curve
        `nearest_neighbour` matches the arc points to the nearest bezier curve points

    extra_per_arch: int
        extra control point per arch (internals, the two bounding are selected based on the arc extremity)
    
    resolution: int (used only when kind="nearest_neighbour")
        number of bezier points to evaluate, if None use the number of points in the arch

    return_filtered_bool: bools
        wether to return a boolean to check which of the points have been kept
    
    Returns
    -------
    points_to_add: np.ndarray
        The new points sorted

    filtered: np.ndarray of bool (only if return_filtered_bool == True)
    """
    
    if return_filtered_bool:
        filtered, arches = make_bounding_arches(points, mask, epsilon, return_filtered_bool=return_filtered_bool)
    else:
        arches = make_bounding_arches(points, mask, epsilon, return_filtered_bool=return_filtered_bool)

    new_control_points = []
    for arc in arches:
        cpf = ControlPointsFitter(kind=fitter_kind, n_ctrl_points=2 + extra_per_arch, resolution=None)
        new_control_points.append(cpf.fit_predict(arc, only_new=True))
    if return_filtered_bool:
        return np.vstack(new_control_points), filtered
    else:
        return np.vstack(new_control_points)
