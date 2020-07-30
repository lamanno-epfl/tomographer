from __future__ import division
import numpy as np
from skimage.color import rgb2gray
from skimage import transform
from skimage import segmentation
from skimage import morphology
from skimage import transform
from scipy.interpolate import interp1d
from scipy import stats
from scipy import sparse
from scipy.misc import toimage
from scipy.spatial.distance import pdist, squareform
from collections import Counter
from typing import *


def distance_point_line(point: Tuple[float, float], line: Tuple[float, float, float]) -> float:
    '''Calculate the distance between a point and a line

    Parameters
    ----------
    point: touple of float
        Coordinates (x, y)  of the point
    line: touple of float
        Parameters (a,b,c) where the line is ax + by + c = 0
        
    Returns
    -------
    distance: float
        Euclidean distance between the point and the line
    '''
    x0, y0 = point
    a, b, c = line
    return np.abs(a * x0 + b * y0 + c) / np.sqrt(a**2 + b**2)


def linear_kernel(D: Union[np.ndarray, float], w: float) -> np.ndarray:
    '''Returns the linear kernel at a point distant D from the max
    
    Parameters
    ----------
    D: ndarray or float
        distance from the center(max) of the kernel
    w: float
        half width of the wavelet (distance from max to zero)
        for d > w the kernel evaluate to 0
        
    Returns
    -------
    K: ndarray or float
        Array containg at K[i,j] the value of the kernel function
        at D[i,j]. For definition of kernel it is 0<=K[i,j]<=1
    
    '''
    X = D / float(w)
    return (np.abs(X) <= 1) * (1 - np.abs(X))


def calculate_projection(angle: float, q: float, X_coords: np.ndarray, Y_coords: np.ndarray, slice_width: float) -> np.ndarray:
    '''Returns a flattened array of the pixels contribution to the point projection
    
    Parameters
    ----------
    angle: float
        The slope of the line the slide is centered upon
    q: float
        The q parameter of the line (y = mx +q) the slide is centered upon
    X_coords: 2-D array
        contains x at pixel (y,x)
    Y_coords: 2-D array
        contains y at pixel (y,x)
    slice_width: float
        the width of a slice
    sparse: bool, default=False
        wether to return a csr sparse matrix
    Returns
    --------
    design_matrix_row: 1-D array
    
    Notes
    -----
    What is returned by this function correspond to a row of the Design/Projection matrix.
    To be more precise this is the raveled form of a matrix that contains:
    in position (r,c) the contribution of the (r,c) pixel to the point projection.
    
    In practice this is calculated with an heuristic function.
    contribution around the line distributes as a linear (triangular) kernel
    the line y = tan(alpha)*x + q
    If plotted would look like a pixelated line (y = tan(alpha) + q) that skips all the
    zero point of the image mask
    
    The reason why this is used and prefered over other kernels is that this is robust
    over summation. This means that summing all the pixel contributions one gets back 
    the original grayscale image mask.
    '''
    # Explict line parameters
    a = np.tan(angle)
    b = -1.
    c = q
    # Distance of every point from the line
    D = distance_point_line((X_coords, Y_coords), (a, b, c))
    # Contribution of every pixel, caluclated with linear kernel heuristic
    M = linear_kernel(D, slice_width)  # NB! the total width of the kernel is duble the slice_widht
    # Return it ravelled
    return M.ravel()


def find_extremities(image: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    '''Returns the first and last point encoutered by slicing
    the mask `image` with angle `angle`
    
    Parameters
    ----------
    image: 2-D array
        the convex_hull of the mask
    angle: float
        (in radiants) the angle of the slicing with respect to x (conter clockwise)
    
    Returns
    -------
    first_point, last_point: arrays
        first and last point touched rotating counterclockwise (clockwise if looking at the image) direction
                            
    Notes
    -----
    Given a slicing angle and a mask it returns the first and
    last point encoutered by the slicing. We use a rotation
    trick to do this in the fastest and most robust way
    
    Timing: 2.75 ms
    '''
    
    # To rotate around the center of the image
    # we compose translation to the center and rotation
    # we retranslate to bring back in position
    shift_y, shift_x = (np.array(image.shape) - 1) / 2.
    tf_rotate = transform.SimilarityTransform(rotation=-angle)
    tf_shift = transform.SimilarityTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = transform.SimilarityTransform(translation=[shift_x, shift_y])
    
    # we only use the inverse so we assign this and not the direct transformation
    tf_composed_inv = (tf_shift + (tf_rotate + tf_shift_inv)).inverse
    
    # Rotate the image and find the max and min point
    image_rotated = transform.warp(image, inverse_map=tf_composed_inv, order=1)  # order might be changed to 3
    # the line above returns a float after transformation even if input was bool
    pos_cordinates = list(zip(*np.where(image_rotated > 0.7)))  # we threshold because of the reason above
    y_min, x_min = min(pos_cordinates)
    y_max, x_max = max(pos_cordinates)
    
    # Apply the inverse transformation to return to the original coordinates
    return tf_composed_inv([x_min, y_min])[0], tf_composed_inv([x_max, y_max])[0]


def slicing_parameters(angle: float, extremity_points: Tuple[np.ndarray, np.ndarray]) -> Tuple[float, Tuple[float, float]]:
    '''Calculates all the parameters necessary for slicing.
    
    Parameters
    ----------
    angle: float
        angle in radiants
    extremity_points: touple of 1-D array
        (first_point, last_point) as it is given from find_extremities()
    
    Returns
    -------
    m, qs: floats
        Parameters for the two tangent lines: y = m*x + qs[0] and y = m*x + qs[1]
            
    TO DO: Cases here r // x and r // y
    '''
    
    # the slope
    m = np.tan(angle)
    # the intercepts q = y - m*x
    q_first = extremity_points[0][1] - m * extremity_points[0][0]
    q_last = extremity_points[1][1] - m * extremity_points[1][0]
    
    return m, (q_first, q_last)


def sum_project(image: np.ndarray, angle: float, slice_width: float, notation_inverse: bool=True) -> np.ndarray:
    '''Utility to get the projection of a greayscale `image` in a direction `angle`
    
    Parameters
    ----------
    image: 2-D array
        Grayscale image. Reppresenting a mask or a signal
    angle: float
        In radiants, the angle of slicing
    slice_width: float
        The width of a slice
    notation_inverse: bool
        the angle is ment from the point of view of the image 
    
    Returns
    -------
    projection: 1-D array
        the sum of all the pixels

    TODO: add suppport with precomputed projection matrix
    
    '''
    img_shape = image.shape
    # Build two arrays containing respectivelly the x and the y coordinates of every pixel
    Y_coords, X_coords = np.mgrid[0:img_shape[0], 0:img_shape[1]]
    raveled_img = image.ravel()

    if notation_inverse:
        angle = np.pi - angle
    
    #  when the q2 - q1 is an integer exactly n slides the loss of precision of np.tan will cause an extra slice to be cut 
    _tan = lambda x: np.round(np.tan(x), decimals=4)
    # Calculate the q parameters of the two lines passing by the extremities
    if (angle % np.pi) > np.pi / 2.:
        x0 = np.max(X_coords) + 0.5
        y0 = np.max(Y_coords) + 0.5
        q1 = y0 - _tan(angle) * x0
        x0 = 0 - 0.5
        y0 = 0 - 0.5
        q2 = y0 - _tan(angle) * x0
        # Calculate the distance between two proj lines passing from the corners of the image
        dist = distance_point_line((x0, y0), (_tan(angle), -1, q1))
    else:
        # q = y0 - m * x0]
        x0 = np.max(X_coords) + 0.5
        y0 = 0 - 0.5
        q2 = y0 - _tan(angle) * x0
        x0 = 0 - 0.5
        y0 = np.max(Y_coords) + 0.5
        q1 = y0 - _tan(angle) * x0

        # Calculate the distance between two proj lines passing from the corners of the image
        dist = distance_point_line((x0, y0), (_tan(angle), -1, q2))
    
    number_of_slices = dist / slice_width
        
    # Calculate the different q parameters of the lines around which the slices are centered on
    # The naive qs are of lines start from the tangent and finishes on the tangent
    q_step = np.abs(q2 - q1) / number_of_slices
    if q1 < q2:
        naive_qs = np.arange(q1, q2 + q_step, q_step)
    else:
        naive_qs = np.arange(q1, q2 - q_step, -q_step)

    # The above is not correct: we want the line in the middle of the one generated above
    # adjusted_qs = (naive_qs[1:] + naive_qs[:-1]) /2.
    adjusted_qs = naive_qs
    
    # Inititalize
    proj_matrix = []
    # Iterate over the slices (parameter q)
    for q in adjusted_qs:
        proj_matrix.append(calculate_projection(angle, q, X_coords, Y_coords, slice_width))
            
    # Transform the list of arrays in a 2-D array
    proj_matrix = np.array(proj_matrix)
    
    # Calculate the pojection as a dot product and return
    return np.dot(proj_matrix, raveled_img[:, None])


def create_connectivity(mask: np.ndarray, kind: str="queen") -> np.ndarray:
    """ Create a connectivity matrix of the pixels in a image

    Args
    ----
    mask: np.2darray
        Square image of side N
    kind: str default 'queen
        The kind of connectivity to apply. Can be: rook, bishop, queen (as in chess)

    Returns
    -------
    connectivity_matrix: np.ndarray
        A connectivity matrix (N^2, N^2) where N is the side of mask
    """

    ll = mask.shape[0]
    ll += 2  # add a 1 pixel margin all around the image to simplify the construction (will be removed as last step)
    cstm_connectivity = np.zeros((ll**2, ll**2))  # initialize empty
    pixel_ixs = np.arange(ll**2)

    # Compute the indexes of the fake edges
    real_pixel_bool = np.ones((ll, ll), dtype=bool)
    real_pixel_bool[0, :] = False
    real_pixel_bool[-1, :] = False
    real_pixel_bool[:, 0] = False
    real_pixel_bool[:, -1] = False
    real_pixel_bool = real_pixel_bool.flat[:]
    real_pixel_ixs = pixel_ixs[real_pixel_bool]

    # Neighbour rule
    if kind == "rook":
        neig_relative_ix_pos = np.array([+1, -1, -ll, ll])
    elif kind == "bishop":
        neig_relative_ix_pos = np.array([-ll + 1, ll + 1, -ll - 1, ll - 1])
    elif kind in ["queen", "king"]:
        neig_relative_ix_pos = np.array([+1, -1, -ll, ll, -ll + 1, ll + 1, -ll - 1, ll - 1])

    # Set True at where there is connectivity
    cstm_connectivity[real_pixel_ixs[:, None], real_pixel_ixs[:, None] + neig_relative_ix_pos] = True
    # Is the same as following but using broadcasting
    # for i in real_pixel_ixs:
    #     cstm_connectivity[i, neig_relative_ix_pos+i] = True

    # Remove connectivity entry corresponding to the dummy 1 pixel edges
    cstm_connectivity = cstm_connectivity[real_pixel_ixs[:, None], real_pixel_ixs[None, :]]
    
    return cstm_connectivity


def place_inside_mask(values: np.ndarray, mask_bw: np.ndarray) -> np.ndarray:
    """Place the values at the position that are 1/True in mask followin the C_CONTIGUOUS enumeration order

    Args
    ----
    values: np.ndarray (1d, float)
        the vaues to fill in mask
    mask_bw: np.ndarray (2d, binary or boolean)
        the mask to fill values in
    Returns
    -------
    x: np.ndarray 2d
        2d array with the values subsittuted in the right place of the mask
    """
    # assert np.allclose(mask.sum(), len(values))
    x = np.zeros(mask_bw.shape)
    x.flat[mask_bw.flat[:].astype(bool)] = values
    return x


def build_Design_Matrix(angles: np.ndarray, widths: List[float],
                        mask_g: np.ndarray, mask_thrs: float=0.2,
                        notation_inverse: bool=True, return_projlen: bool=True,
                        return_sparse: bool=False) -> np.ndarray:
    '''Builds the regression design matrix (Projection Matrix).
    
    Parameters
    ----------
    angles: np.ndarray
        the angles of the slicing
    widths: list of float
        (number of pixels) real width of a slice for every cutting angle
    mask_g: 2-D array of floats
        greyscale mask reppresenting the shape of the tissue slice,
        works good also if entries are only 1s and 0s
    mask_thrs: float
        value to threshold mask_g
    notation_inverse: bool, default=True
        the angles are ment from the point of view of the image
    return_projlen: bool, default=True
        wether to return the information about the number of rows for each angle
    return_sparse: bool, default=False
        wether to return a scipy.sparse.csr_matrix
    
    Returns
    -------
    design_matrix: 2-d array
    
    Notes
    -----
    it returns design matrix for the mask constrained regression problem
    the correct angles would be the one looking at the image flipped (origin at left-top positive of y is down)
    but if notation_inverse == True it assumes that angles are respect to a origin at the left-bottom
    
    Assumptions: The image is reliable and the width are reliable. 
                This has to be adjusted beforehand
    '''
    if notation_inverse:
        angles = (np.pi - angles) % (2 * np.pi)

    img_shape = mask_g.shape
    # Build two arrays containing respectivelly the x and the y coordinates of every pixel
    Y_coords, X_coords = np.mgrid[0:img_shape[0], 0:img_shape[1]]
    # Prepare the raveled image for the multiplication below
    raveled_img = mask_g.ravel()
    
    # Initialize a list to which the rows of the design matrix will be appended
    Z = []
    projlen = []
    # Iterate over the angles of projections
    for n_a, angle in enumerate(angles):
        
        # Calculate q1 and q2
        conv_hull = morphology.convex_hull_image(mask_g > mask_thrs)  # Note: before 0.05
        first_point, last_point, q1, q2, extension = all_slicing_parameters(conv_hull, angle)

        # Calculate the distance between the first and last line
        number_of_slices = extension / widths[n_a]
        
        # Calculate the different q parameters of the lines around which the slices are centered on
        # The naive qs are of lines start from the tangent and finishes on the tangent
        q_step = np.abs(q2 - q1) / number_of_slices
        if q1 < q2:
            naive_qs = np.arange(q1, q2 + q_step, q_step)
        else:
            naive_qs = np.arange(q1, q2 - q_step, -q_step)

        # This is not correct and we want the line in the middle of the one generated above
        # adjusted_qs = (naive_qs[1:] + naive_qs[:-1]) /2.
        adjusted_qs = naive_qs
        projlen.append(len(adjusted_qs))
        # Iterate over the slices (over q)
        for q in adjusted_qs:
            # Calculate the row of the design matrix and append it
            if return_sparse:
                Z.append(sparse.csr_matrix(calculate_projection(angle, q, X_coords, Y_coords, widths[n_a]) * raveled_img))
            else:
                Z.append(calculate_projection(angle, q, X_coords, Y_coords, widths[n_a]) * raveled_img)
            
    # transform the list of arrays in a 2-D array
    if return_sparse:
        if return_projlen:
            return sparse.vstack(Z), projlen
        return sparse.vstack(Z)
    else:
        if return_projlen:
            return np.array(Z), projlen
        return np.array(Z)


def all_slicing_parameters(convex_hull_mask: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    m = np.tan(angle)  # Don't round, the fact that the maximum yielded is 1e16 is good, avoids nan and good precision
    yy, xx = np.where(convex_hull_mask)
    qq = yy - m * xx
    q1 = np.max(qq)
    q2 = np.min(qq)
    ix_max, ix_min = qq.argmax(), qq.argmin()
    first_point = (xx[ix_max], yy[ix_max])
    last_point = (xx[ix_min], yy[ix_min])
    last_line = (-np.tan(angle), 1., -q2)
    extension = distance_point_line(first_point, last_line)  # this function is nan-proof when use np.tan
    if 0.5 * np.pi < (angle % (2 * np.pi)) <= 1.5 * np.pi:
        return first_point, last_point, q1, q2, extension
    else:
        return last_point, first_point, q2, q1, extension


def prepare_design_symmetry(D: np.ndarray) -> np.ndarray:
    """Add to the design matrix strip simmetric to the one in the input
    
    Args
    ----
    D: np.ndarray
        The design matrix as it comes from build_Design_Matrix

    Returns
    -------
    New design matrix with appended the simmetric angles

    """
    # Flip the mask assuming simmetry
    MD, ND = D.shape
    sqrtND = int(np.sqrt(ND))
    cols_ord = np.arange(ND).reshape(sqrtND, sqrtND)[:, ::-1].ravel()  # flipped index
    return np.append(D, D[:, cols_ord], 0)


def prepare_design_masked(D: np.ndarray, mask_bw: np.ndarray) -> np.ndarray:
    """Design matrix using only the pixels in the mask
    
    Args
    ----
    D: np.ndarray
        The design matrix as it comes from build_Design_Matrix
    mask_bw: np.ndarray (binary or boolean)
        The image used as mask. Entryes should be only 1/0 or True/False

    Returns
    -------
    New design matrix with appended the simmetric angles. And where only the pixels in the mask are considered

    """
    filt_mask = mask_bw.flat[:].astype(bool)  # prepare the mask to be used as a boolean filter
    return np.copy(D[:, filt_mask])


def prepare_design_symmetry_masked(D: np.ndarray, mask_bw: np.ndarray) -> np.ndarray:
    """Design matrix using only the pixels in the mask and adding strips simmetric to the one in the input
    
    Args
    ----
    D: np.ndarray
        The design matrix as it comes from build_Design_Matrix
    mask_bw: np.ndarray (binary or boolean)
        The image used as mask. Entryes should be only 1/0 or True/False

    Returns
    -------
    New design matrix with appended the simmetric angles. And where only the pixels in the mask are considered

    """
    filt_mask = mask_bw.flat[:].astype(bool)  # prepare the mask to be used as a boolean filter
    return prepare_design_symmetry(D)[:, filt_mask]


def prepare_observations(projections: List[np.ndarray], xs: List[np.ndarray],
                         first_points: List[int], projs_len: List[int],
                         interpolation: str="linear", verbose: bool=False) -> np.ndarray:
    """Prepare the observation vector `b`
    
    Args
    ----------
    projections: list of 1-D array
        a list of projection. Projections are first selected so that the first value is the first reliable section and
        the last the last reliable section
    xs: list of 1-D array
        Contains arrays indicating wich are indexes of the 'projections' input.
        `projections` are usually filtered and trimmed, so an ix array is kept to keep track of it. 
        Its values[i] usually gets filtered and some samples are missing.
        e.g. [ array([20,21,22,24,25,27,30,...]), array([10,11,12,15,16,17,18,...]), array([3,4,7,9,10,11,12,...]) ]
    first_points: list of int
        for every proj-angle it indicates how shifted it is from the theoretical one
    projs_len: list of int
        the expected number of slices that should be taken in account starting from `list_of_points[i]`
    interpolation: str
        kind interpolation one of "linear", "cubic", "mixed"
    verbose: bool
        prints min(xi), max(xi), max(xi)-min(xi)+1, n_points, len(xi), len(p)

    Returns
    -------
    final_proj: 1-D array
        The projections ready to be given as an imput of a regression problem
    """

    final_proj = np.array([])
    for projection, xi, first_point, n_points in zip(projections, xs, first_points, projs_len):
        full_x = np.arange(first_point, first_point + n_points)
        p = projection.copy()
        
        # take the points between first point and last point
        bool_ix = (first_point <= xi) & (xi < first_point + n_points)
        xi = xi[bool_ix]
        p = p[bool_ix]

        # Deal with some special cases if they occur (e.g. samples at the extremities did not wokr)
        if first_point not in xi:
            xi = np.r_[first_point, xi]
            p = np.r_[0, p]
        if (first_point + n_points - 1) not in xi:
            xi = np.r_[xi, first_point + n_points - 1]
            p = np.r_[p, 0]
            
        if verbose:
            print(np.min(xi), np.max(xi), np.max(xi) - np.min(xi) + 1, n_points, len(xi), len(p))
        
        # Perform the interpolation for the missing ixs
        if interpolation == "linear":
            f1 = interp1d(xi, p, kind='linear', fill_value=0, bounds_error=False)
            interpolated = f1(full_x)
        elif interpolation == "cubic":
            f3 = interp1d(xi, p, kind='cubic', fill_value=0, bounds_error=False)
            interpolated = np.clip(f3(full_x), a_min=0, a_max=1.2 * max(p))
        elif interpolation == "mixed":
            f1 = interp1d(xi, p, kind='linear', fill_value=0, bounds_error=False)
            f3 = interp1d(xi, p, kind='cubic', fill_value=0, bounds_error=False)
            intp1 = np.clip(f1(full_x), a_min=0, a_max=1.2 * max(p))
            intp3 = np.clip(f3(full_x), a_min=0, a_max=1.2 * max(p))
            intp0 = intp1 * np.array([(i in xi) for i in full_x])
            interpolated = 0.15 * intp3 + 0.35 * intp1 + 0.5 * intp0

        final_proj = np.r_[final_proj, interpolated]  # This is just appending to the previous projections
    return final_proj


def prepare_observations_symmetry(projections: List[np.ndarray], xs: List[np.ndarray],
                                  first_points: List[int], projs_len: List[int],
                                  interpolation: str="linear", verbose: bool=False) -> np.ndarray:
    """Prepare the observation vector `b` assuming symmetry.
    It will will copy the observations at one angle so to assume that the projection at the symmetrical is the same
    
    Args
    ----------
    projections: list of 1-D array
        a list of projection. Projections are first selected so that the first value is the first reliable section and
        the last the last reliable section
    xs: list of 1-D array
        Contains arrays indicating wich are indexes of the 'projections' input.
        `projections` are usually filtered and trimmed, so an ix array is kept to keep track of it.
        Its values[i] usually gets filtered and some samples are missing.
        e.g. [ array([20,21,22,24,25,27,30,...]), array([10,11,12,15,16,17,18,...]), array([3,4,7,9,10,11,12,...]) ]
    first_points: list of int
        for every proj-angle it indicates how shifted it is from the theoretical one
    projs_len: list of int
        the expected number of slices that should be taken in account starting from `list_of_points[i]`
    interpolation: str
        kind interpolation one of "linear", "cubic", "mixed"
    verbose: bool
        prints min(xi), max(xi), max(xi)-min(xi)+1, n_points, len(xi), len(p)

    Returns
    -------
    final_proj: 1-D array
        The projections ready to be given as an imput of a regression problem
    """

    final_proj = prepare_observations(projections, xs, first_points, projs_len, interpolation, verbose)
    final_proj = np.r_[final_proj, final_proj]
    return final_proj


def prepare_regression(projections: List[np.ndarray], xs: List[np.ndarray], design_matrix: np.ndarray,
                       first_points: List[int], projs_len: List[int], verbose: bool=True) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Prepare Design matrix and observation vector

    Parameters
    ----------
    projections: list of 1-D array
        a list of projection. Projections are first selected so that the first value is the first reliable section and
        the last the last reliable section
    xs: list of 1-D array
        Contains arrays indicating wich are indexes of the 'projections' input.
        `projections` are usually filtered and trimmed, so an ix array is kept to keep track of it. 
        Its values[i] usually gets filtered and some samples are missing.
        e.g. [ array([20,21,22,24,25,27,30,...]), array([10,11,12,15,16,17,18,...]), array([3,4,7,9,10,11,12,...]) ]
    design_matrix: 2-D array
        as calculated by the fucntion build_Design_Matrix
    first_points: list of int
        for every proj-angle it indicates how shifted it is from the theoretical one
    projs_len: list of int
        the expected number of slices that should be taken in account starting from `list_of_points[i]`
    verbose: bool
        prints min(xi), max(xi), max(xi)-min(xi)+1, n_points, len(xi), len(p)

    Returns
    -------
    D: 2-D array
        The design matrix ready to be given as as a input of a regression problem
    final_proj: 1-D array
        The projections ready to be given as an imput of a regression problem

    Notes
    -----
    The function includes a mixed cubic-linear-zero interpolation to fill in the missing values. 
    This is necessary because if one instead omits the equations for the missing projection (as it would be intuitive)
    the regularized problem will have less constrains and the respective pixel will be set 
    to zero or very low numbers.
    Input image given to Design Matrix function has to be symmetrical.
    '''
    D = design_matrix.copy()
    final_proj = prepare_observations(projections, xs, first_points, projs_len, verbose=verbose)
    return D, final_proj


def prepare_regression_symmetry(projections: List[np.ndarray], xs: List[np.ndarray], design_matrix: np.ndarray, first_points: List[int]=[0,0,0], projs_len: List[int]=[100,100,100], verbose: bool=True) -> Tuple[np.ndarray, np.ndarray]:
    '''Currently the best algorythm for reconstruction explointing the simmetry of the input image.

    Parameters
    ----------
    projections: list of 1-D array
        a list of projection. Projections are first selected so that the first value is the first reliable section and
        the last the last reliable section
    xs: list of 1-D array
        Contains arrays indicating wich are indexes of the 'projections' input.
        `projections` are usually filtered and trimmed, so an ix array is kept to keep track of it.
        Its values[i] usually gets filtered and some samples are missing.
        e.g. [ array([20,21,22,24,25,27,30,...]), array([10,11,12,15,16,17,18,...]), array([3,4,7,9,10,11,12,...]) ]
    design_matrix: 2-D array
        as calculated by the fucntion build_Design_Matrix
    first_points: list of int
        for every proj-angle it indicates how shifted it is from the theoretical one
    projs_len: list of int
        the expected number of slices that should be taken in account starting from `list_of_points[i]`
    verbose: bool
        prints min(xi), max(xi), max(xi)-min(xi)+1, n_points, len(xi), len(p)

    Returns
    -------
    D: 2-D array
        The design matrix ready to be given as as a input of a regression problem
    final_proj: 1-D array
        The projections ready to be given as an imput of a regression problem

    Notes
    -----
    The function includes a mixed cubic-linear-zero interpolation to fill in the missing values. 
    This is necessary because if one instead omits the equations for the missing projection (as it would be intuitive)
    the regularized problem will have less constrained and the respective pixel will be set 
    to zero or very low numbers.
    Input image given to Design Matrix function has to be symmetrical.
    '''

    D = design_matrix.copy()
    D = prepare_design_symmetry(D)
    final_proj = prepare_observations_symmetry(projections, xs, first_points, projs_len, verbose=verbose)
    return D, final_proj


def prepare_regression_symmetry_masked(projections: List[np.ndarray], xs: List[np.ndarray],
                                       design_matrix: np.ndarray, mask: np.ndarray,
                                       first_points: List[int], projs_len: List[int],
                                       verbose: bool=True) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Currently the best and faster algorythm for reconstruction explointing the simmetry of the input image.

    Parameters
    ----------
    projections: list of 1-D array
        a list of projection. Projections are first selected so that the first value is the first reliable section and
        the last the last reliable section
    xs: list of 1-D array
        Contains arrays indicating wich are indexes of the 'projections' input.
        `projections` are usually filtered and trimmed, so an ix array is kept to keep track of it.
        Its values[i] usually gets filtered and some samples are missing.
        e.g. [ array([20,21,22,24,25,27,30,...]), array([10,11,12,15,16,17,18,...]), array([3,4,7,9,10,11,12,...]) ]
    design_matrix: 2-D array
        as calculated by the fucntion build_Design_Matrix
    mask: 2-D boolean array:
        a boolean masked indicating which pixel to reconstruct
    first_points: list of int
        for every proj-angle it indicates how shifted it is from the theoretical one
    projs_len: list of int
        the expected number of slices that should be taken in account starting from `list_of_points[i]`
    verbose: bool
        prints min(xi), max(xi), max(xi)-min(xi)+1, n_points, len(xi), len(p)

    Returns
    -------
    D: 2-D array
        The design matrix ready to be given as as a input of a regression problem
    final_proj: 1-D array
        The projections ready to be given as an imput of a regression problem

    Notes
    -----
    The function includes a mixed cubic-linear-zero interpolation to fill in the missing values. 
    This is necessary because if one instead omits the equations for the missing projection (as it would be intuitive)
    the regularized problem will have less constrained and the respective pixel will be set 
    to zero or very low numbers.
    Input image given to Design Matrix function has to be symmetrical.
    '''

    D = prepare_design_symmetry_masked(D, mask)
    final_proj = prepare_observations_symmetry(projections, xs, first_points, projs_len, verbose=verbose)
    return D, final_proj


# From the images Simone took I calculated the the pixel/mm ratio
# using imageJ and the known distance of the mold
# 
# 46.35pixels/mm
# for the file
# plateMean_finalmask.png
# 
# Using this pixel ratio I extimate the brain slice size
# MaxDiameter = 8.3 mm
# minDiameter = 4.9 mm