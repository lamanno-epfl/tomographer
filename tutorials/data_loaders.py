import numpy as np
import pandas as pd
import h5py
import re
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform

from sklearn.cluster import DBSCAN, KMeans, SpectralClustering, AgglomerativeClustering, ward_tree


def retrieve_idx(selected_genes, keyword):
    # Display genes that are found
    keyword = keyword.lower()
    print('\nThese are the genes that fit the query:\n')
    for i, gene in enumerate(selected_genes):
        if keyword in gene.lower():
            print(i, gene)
            
    input_gene = input('Input idx (displayed beside gene)')
    return int(input_gene)

def display_gene(gene_idx, X, mask, selected_genes, with_mouse=True):
    """Display the gene given the idx and the matrix that must be queried
    Can show the mouse homolog if desired"""
    plot = np.reshape(X[:, gene_idx], (mask.shape[0], mask.shape[1]))
    
    if with_mouse:
        gene_mouse_name = input('Name of Mouse Homolog (case sensitive):')
        plt.subplot(121)
        plt.imshow(plot)
        plt.subplot(122)
        create_mouse_plot(gene_mouse_name, mask2D)
        plt.suptitle(selected_genes[gene_idx])
    else: 
        plt.imshow(plot)
        plt.title(selected_genes[gene_idx])
        plt.show
        
def get_mouse_layout(gene, mask3D):
    obj = type('namespace', (object,), {})
    var = dict()
    var[gene] = obj()
    var[gene].Rgt3D = shift_simmetrize(pad_to_cube(ish[gene][:]), 1) * mask3D
    var[gene].Rgt2D = var[gene].Rgt3D[33,:,:]
    return var[gene].Rgt2D

def create_mouse_plot(gene, mask):
    mouse_plot = get_mouse_layout(gene, mask3D)
    xmin, xmax, ymin, ymax = np.amin(mouse_plot), np.amax(mouse_plot), np.amin(mouse_plot), np.amax(mouse_plot)
    extent = xmin, xmax, ymin, ymax
    masked = plt.imshow(mask, extent=extent)
    gene_expression = plt.imshow(mouse_plot, extent=extent, cmap=plt.cm.viridis, alpha=0.9)
    
    
def generate_template(mask, mask_idx_1D, idx):
    img_tmp = np.zeros((mask.shape[0], mask.shape[1])).flatten()
    img_tmp[mask_idx_1D[idx]] = 1
    return np.reshape(img_tmp, (mask.shape[0], mask.shape[1]))

def get_coordinate(img_tmp):
    y_coord = []
    x_coord = []
    for idx, row in enumerate(img_tmp):
        for idx2, m in enumerate(row):
            if m != 0:
                y_coord.append(idx)
                x_coord.append(idx2)
    return np.array(y_coord), np.array(x_coord)

def get_contour_values(shaded_matrix):
    """Find coordinates of pixels on the border of a shaded area"""
    contour_vals = []
    for i, row in enumerate(shaded_matrix):
        if max(row) == 1:
            idx_1 = (np.argwhere(shaded_matrix[i] == 1))
            contour_vals.append((i, min(idx_1)[0]))
            contour_vals.append((i, max(idx_1)[0]))
    for i, col in enumerate(shaded_matrix.T):
        if max(col) == 1:
            idx_1 = (np.argwhere(shaded_matrix.T[i] == 1))
            contour_vals.append((min(idx_1)[0], i))
            contour_vals.append((max(idx_1)[0], i))
    
    return contour_vals


def plot_contour(contour_vals, mask):
    """Extract only the values of the contours"""
    contour_2D = np.zeros((mask.shape[0], mask.shape[1]))
    for pixel_position in contour_vals:
        contour_2D[pixel_position] = 1
    return contour_2D

def gene_profile_error(gene, bins=20):
    """Detect genes that were reconstructed poorly based on their pixel value frequencies"""
    x = np.histogram(gene, bins=bins) # get frequencies
    cleaned_vec = np.log(x[0]).copy()
    cleaned_vec[np.isneginf(cleaned_vec)] = 0 # remove negative infinity value, set to 0
    
    x_range = len(x[1]) - 1 # set the number of x values
    
    # fit linear model
    fit = np.polyfit(np.arange(x_range), cleaned_vec, 1)
    p_ = np.poly1d(fit)
    xp = np.arange(x_range).copy()
    
    return abs(p_(xp) - cleaned_vec).sum() ** 2 # return MSE

def generate_region_mask(region_dict):
    region_mask = np.zeros((mask.shape[0], mask.shape[1]))
    for coordinate in np.array(region_dict):
        region_mask[coordinate[1]][coordinate[0]] = 1
        region_mask[coordinate[1]][mask.shape[1] - coordinate[0]] = 1
    return region_mask

def retrieve_correlated_genes(X_filtered, region_mask):
    """given X and a regional mask, return a list of the most similar genes ranked"""
    correlations = []
    for i, col in enumerate(np.arange(X_filtered.shape[1])):
        correlations.append((np.corrcoef(X_filtered[:,col],
                                         region_mask.flatten())[0,1], selected_genes_filtered[i], i))
    correlations.sort(key = operator.itemgetter(0), reverse=True)
    
    return correlations

def generate_template_subcluster(mask, mask_idx_1D, idx, ix):
    """will provide values for a subsection of the mask"""
    img_tmp = np.zeros((mask.shape[0], mask.shape[1])).flatten()
    img_tmp[mask_idx_1D[idx[ix]]] = 1
    return np.reshape(img_tmp, (mask.shape[0], mask.shape[1]))

def get_subcluster_coordinates(subclass, classes_umap, embedding, clusters=6):

    class_ = np.argwhere(classes_umap == subclass)
    class_ = np.reshape(embedding[class_], (len(class_), 2))

    agg_umap_ = AgglomerativeClustering(n_clusters=6)
    subclasses_ = agg_umap_.fit_predict(class_)

    # Project clusters onto mask
    class_ = np.argwhere(classes_umap == subclass)
    subregion_dict = {}
    for i in np.arange(clusters):
        ix = np.argwhere(subclasses_ == i)
        img_tmp = generate_template_subcluster(mask, mask_idx_1D, class_, ix)
        idx, idx2 = get_coordinate(img_tmp)
        subregion_dict[i] = list(zip(idx2, idx))
        #plt.scatter(idx2, idx, c=colors[i], s=10)
    return subregion_dict



from numpy import *
import getopt
import sys

class Results:
        pass

def calc_loccenter(x, lin_log_flag):
    M,N = x.shape
    if N==1 and M>1:
        x = x.T
    M,N = x.shape
    loc_center = zeros(M)
    min_x = x.min(1)
    x = x - min_x[:,newaxis]
    for i in range(M):
        ind = where(x[i,:]>0)[0]
        if len(ind) != 0:
            if lin_log_flag == 1:
                w = x[i,ind]/sum(x[i,ind], 0)
            else:
                w = (2**x[i,ind])/sum(2**x[i,ind], 0)
            loc_center[i] = sum(w*ind, 0)       
        else:
            loc_center[i] = 0

    return loc_center

def _calc_weights_matrix(mat_size, wid):
    '''Calculate Weight Matrix
    Parameters
    ----------
    mat_size: int
        dimension of the distance matrix
    wid: int
        parameter that controls the width of the neighbourood
    Returns
    -------
    weights_mat: 2-D array
        the weights matrix to multiply with the distance matrix
    '''
    #calculate square distance from the diagonal
    sqd = (arange(1,mat_size+1)[newaxis,:] - arange(1,mat_size+1)[:,newaxis])**2
    #make the distance relative to the mat_size
    norm_sqd = sqd/wid
    #evaluate a normal pdf
    weights_mat = exp(-norm_sqd/mat_size)
    #avoid useless precision that would slow down the matrix multiplication
    weights_mat -= 1e-6
    weights_mat[weights_mat<0] = 0
    #normalize row and column sum
    weights_mat /= sum(weights_mat,0)[newaxis,:]
    weights_mat /= sum(weights_mat,1)[:, newaxis]
    #fix asimmetries
    weights_mat = (weights_mat + weights_mat.T) / 2.
    return weights_mat


def _sort_neighbourhood( dist_matrix, wid ):
    '''Perform a single iteration of SPIN
    Parameters
    ----------
    dist_matrix: 2-D array
        distance matrix
    wid: int
        parameter that controls the width of the neighbourood
    Returns
    -------
    sorted_ind: 1-D array
        indexes that order the matrix
    '''
    assert wid > 0, 'Parameter wid < 0 is not allowed'
    mat_size = dist_matrix.shape[0]
    #assert mat_size>2, 'Matrix is too small to be sorted'
    weights_mat = _calc_weights_matrix(mat_size, wid)
    #Calculate the dot product (can be very slow for big mat_size)
    mismatch_score = dot(dist_matrix, weights_mat)
    energy, target_permutation = mismatch_score.min(1), mismatch_score.argmin(1)
    max_energy = max(energy)
    #Avoid points that have the same target_permutation value
    sort_score = target_permutation - 0.1 * sign( (mat_size/2 - target_permutation) ) * energy/max_energy
    #sort_score = target_permutation - 0.1 * sign( 1-2*(int(1000*energy/max_energy) % 2) ) * energy/max_energy # Alternative
    # Sorting the matrix
    sorted_ind = sort_score.argsort(0)[::-1]
    return sorted_ind


def sort_mat_by_neighborhood(dist_matrix, wid, times):
    '''Perform several iterations of SPIN using a fixed wid parameter
    Parameters
    ----------
    dist_matrix: 2-D array
        distance matrix
    wid: int
        parameter that controls the width of the neighbourood
    times: int
        number of repetitions
    verbose: bool
        print the progress
    Returns
    -------
    indexes: 1-D array
        indexes that order the matrix
    '''
    # original indexes
    indexes = arange(dist_matrix.shape[0])
    for i in range(times):
        #sort the sitance matrix according the previous iteration
        tmpmat = dist_matrix[indexes,:] 
        tmpmat = tmpmat[:,indexes]
        sorted_ind = _sort_neighbourhood(tmpmat, wid);
        #resort the original indexes
        indexes = indexes[sorted_ind]
    return indexes


def gene_profile_error(gene, bins=20):
    """Detect genes that were reconstructed poorly based on their pixel value frequencies"""
    x = np.histogram(gene, bins=bins) # get frequencies
    cleaned_vec = np.log(x[0]).copy()
    cleaned_vec[np.isneginf(cleaned_vec)] = 0 # remove negative infinity value, set to 0
    
    x_range = len(x[1]) - 1 # set the number of x values
    
    # fit linear model
    fit = np.polyfit(np.arange(x_range), cleaned_vec, 1)
    p_ = np.poly1d(fit)
    xp = np.arange(x_range).copy()
    
    return abs(p_(xp) - cleaned_vec).sum() ** 2 # return MSE

def access_idx(image_xy, mask):
    """function to find the index value in the R matrix that corresponds to pixel x,y in the mask"""
    return image_xy[0] * mask.shape[1] + image_xy[1]

def laplacian(X_i, mask):
    """Get vector that corresponds to the coefficients that the kernel dots with the values in R"""
    l_ = X_i[1] - 1
    r_ = X_i[1] + 1
    u_ = X_i[0] - 1
    d_ = X_i[0] + 1
    
    # Find the indexes for the flattened array
    ul = access_idx((u_,l_))
    l = access_idx((X_i[0], l_))
    dl = access_idx((d_, l_))
    u = access_idx((u_, X_i[1]))
    d = access_idx((d_, X_i[1]))
    ur = access_idx((u_, r_))
    r = access_idx((X_i[0], r_))
    dr = access_idx((d_, r_))
    
    # create vector corresponding to kernel
    vector = np.zeros(mask.shape[0] * mask.shape[1])
    
    # set dimension
#     vector[ul] = -1
    vector[l] = -1
#     vector[dl] = -1
#     vector[ur] = -1
    vector[r] = -1
#     vector[dr] = -1
    vector[u] = -1
    vector[d] = -1
    vector[access_idx(X_i)] = 4

    return vector



def set_graph(mask, sigma=10):
    mask_idx = np.argwhere(mask >= 0.2)
    nn = NearestNeighbors()
    nn.fit(mask_idx)
    graph = nn.kneighbors_graph(mask_idx, n_neighbors=50, mode="distance")
    w = 1 / (np.sqrt(2*np.pi) * sigma )* np.exp(-graph.data**2/(2*sigma))
    graph.data = w
    graph = graph.multiply(graph.sum(1).max() / graph.sum(1).A.flat[:][:, None])
    
    return graph




def create_X(mask, genes, alpha_betas, gene_file, graph, threshold=6):
    # Creating matrix X; rows are pixels and features are genes
    pixels = mask.shape[0]**2

    X = np.zeros((pixels, len(genes)), dtype="float64")
    for i, gene in enumerate(genes):
        beta = alpha_betas[gene][:][1]
        if beta > threshold:
            genes.remove(gene)
            continue
        img = gene_file[gene][:]
        # img = gaussian_filter(img,sigma=2, truncate=6)
        img = 0.5*(img + img[:,::-1])
        img = graph.dot(img[np.where(mask >= 0.2)])
        X[np.where(mask.flatten() >= 0.2),i] = img.flat[:]
    X = np.clip(X,0,np.inf)
    X = X[:,:len(genes)]
    # Removing artifacts
    antimask = 1 - mask
    #no_artifact = (mask.flat[:][:,None]*X).sum(0) > (antimask.flat[:][:,None]*X).sum(0)
    #X = X[:,no_artifact] # select only features that have no artifacts
    #selected_genes = np.array(genes)[no_artifact] # corresponding gene array
    selected_genes = np.array(genes)
    return X, selected_genes


def normalize(genes, input_genes, X, verbose=False):
    normalizing_factor = []

    for i, gene in enumerate(genes):
        total_gene_proj_count = 0
        for angle in list(input_genes['genes'][gene]):
            total_gene_proj_count += sum(list(input_genes['genes'][gene][angle]))
        image_val_sum = X[:,i].sum()
        normalizing_factor.append(total_gene_proj_count / image_val_sum)
        if verbose:
            if i % 1000 == 0:
                print('{}% Done'.format(int(i/len(genes) * 100)))

    # Update values in X_normalized
    X_normalized = X.copy()
    for i in range(X_normalized.shape[1]):
        X_normalized.T[i] = X_normalized.T[i] * normalizing_factor[i]
    
    return X_normalized

def SPIN(dist_mat, wid, iters=40, random_state=10):
    np.random.seed(random_state)
    images = []
    ix = sort_mat_by_neighborhood(dist_mat, wid[0], iters)
    sorted_index = ix.copy()
    tmp = dist_mat[ix]
    tmp = tmp[:, ix]
    images.append(tmp)
    for w in wid[1:]:
        print(f'wid: {w}')
        ix_ = sort_mat_by_neighborhood(tmp, w, iters)
        sorted_index = sorted_index[ix_]
        tmp = tmp[ix_]
        tmp = tmp[:, ix_]
        images.append(tmp)
    return tmp, images, sorted_index

def create_dist_mat(thresholds, X, mask_idx_1D):
    for i, thresh in enumerate(thresholds):
        X_normalized_filtered = X.T[:thresh].copy().T
        X_mask = X_normalized_filtered[mask_idx_1D, :].squeeze()

        if i == 0:
            dist_mat = squareform(pdist(X_mask, metric='correlation'))
        else:
            dist_mat += squareform(pdist(X_mask, metric='correlation'))

    dist_mat = dist_mat / len(thresholds) # averaging
    return dist_mat
