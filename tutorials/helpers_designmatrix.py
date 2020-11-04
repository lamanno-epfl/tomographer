import numpy as np


def find_index(mask_idx, b_centers, index):
    tmp1 = np.argwhere(mask_idx[:,0] == b_centers[index][::-1][0]).flatten()
    tmp2 = np.argwhere(mask_idx[:,1] == b_centers[index][::-1][1]).flatten()
    
    # calculate intersection
    intersection = set(tmp1).intersection(tmp2)

    return list(intersection)[0]


def find_index_neighbors(mask_idx, neighbors):

    indexes = []
    for i, n in enumerate(neighbors):
        tmp1 = np.argwhere(mask_idx[:,0] == n[0]).flatten()
        tmp2 = np.argwhere(mask_idx[:,1] == n[1]).flatten()

        intersection = set(tmp1).intersection(tmp2)

        indexes.append(list(intersection)[0])
    return np.array(indexes)

CSFCof  = np.asarray([1.608443, 2.339554, 2.573509, 1.608443, 1.072295, 0.643377, 0.504610, 0.421887,
           2.144591, 2.144591, 1.838221, 1.354478, 0.989811, 0.443708, 0.428918, 0.467911,
           1.838221, 1.979622, 1.608443, 1.072295, 0.643377, 0.451493, 0.372972, 0.459555,
           1.838221, 1.513829, 1.169777, 0.887417, 0.504610, 0.295806, 0.321689, 0.415082,
           1.429727, 1.169777, 0.695543, 0.459555, 0.378457, 0.236102, 0.249855, 0.334222,
           1.072295, 0.735288, 0.467911, 0.402111, 0.317717, 0.247453, 0.227744, 0.279729,
           0.525206, 0.402111, 0.329937, 0.295806, 0.249855, 0.212687, 0.214459, 0.254803,
           0.357432, 0.279729, 0.270896, 0.262603, 0.229778, 0.257351, 0.249855, 0.259950]).reshape(8, 8)

MaskCof = np.asarray([0.390625, 0.826446, 1.000000, 0.390625, 0.173611, 0.062500, 0.038447, 0.026874,
           0.694444, 0.694444, 0.510204, 0.277008, 0.147929, 0.029727, 0.027778, 0.033058,
           0.510204, 0.591716, 0.390625, 0.173611, 0.062500, 0.030779, 0.021004, 0.031888,
           0.510204, 0.346021, 0.206612, 0.118906, 0.038447, 0.013212, 0.015625, 0.026015,
           0.308642, 0.206612, 0.073046, 0.031888, 0.021626, 0.008417, 0.009426, 0.016866,
           0.173611, 0.081633, 0.033058, 0.024414, 0.015242, 0.009246, 0.007831, 0.011815,
           0.041649, 0.024414, 0.016437, 0.013212, 0.009426, 0.006830, 0.006944, 0.009803,
           0.019290, 0.011815, 0.011080, 0.010412, 0.007972, 0.010000, 0.009426, 0.010203]).reshape(8, 8)

def vari(X):
    X = list(X.reshape(-1, 1))
    return (np.var(X)*len(X))

def maskeff(z, zdct, MaskCof):
    '''
    uses MaskCof which is a hardcoded set of values 
    
    '''
    m = 0
    for k in range(8):
        for l in range(8):
            if (k!=0 or l!=0):
                m = m + (zdct[k][l]**2)*MaskCof[k][l]
    papi = vari(z)
    if (abs(papi) > 0):
        papi = (vari(z[0:4,0:4])+vari(z[0:4,4:8])+vari(z[4:8,4:8])+vari(z[4:8,0:4]))/papi
    return (math.sqrt(m*papi)/32)

def psnr_hvs_m(img1, img2, CSFCof):
    
    S1 = 0
    S2 = 0
    height, width = img2.shape
    Num = height*width

    #print ( height, width )
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            A = img1[i:i+8, j:j+8]#.reshape(8, 8)
            B = img2[i:i+8, j:j+8]#.reshape(8, 8)
            A_dct, B_dct = dct(dct(A, norm = 'ortho').T, norm = 'ortho').T, dct(dct(B, norm = 'ortho').T, norm = 'ortho').T
            maskA = maskeff(A, A_dct, MaskCof)
            maskB = maskeff(B, B_dct, MaskCof)
            if (maskB > maskA):
                maskA = maskB
            for k in range(8):
                for l in range (8):
                    u = abs(A_dct[k][l] - B_dct[k][l])
                    S2 = S2 + (u*CSFCof[k][l])**2                 ## this is the PSNR-HVS value
                    if (k!=0 or l!=0):
                        if u < maskA/MaskCof[k][l]:
                            u = 0
                        else:
                            u = u - maskA/MaskCof[k][l]
                    S1 = S1 + (u*CSFCof[k][l])**2                 ## this is the PSNR-HVS-M value
    S1 = S1/Num
    S2 = S2/Num
    if S1 == 0 :
        p_hvs_m = 100000; # img1 and img2 are visually undistingwished
    else:
        p_hvs_m = 10*math.log10(255*255/S1)
    if S2 == 0:  
        p_hvs = 100000    # img1 and img2 are identical
    else:
        p_hvs = 10*math.log10(255*255/S2)
    
    return (p_hvs_m, p_hvs)



def psnr_hvs_maximize(ground_truth, reconstruction):
    """Function takes in as parameters ground truth image and the reconstructed image. Images are not necessarily of the same scale.
    Optimization of the scaling parameter is performed within the function.
    
    Returns maximized PSNR"""

    def maximize_s(s):
        psnr = psnr_hvs_m(ground_truth, s * reconstruction, CSFCof)[0] * -1
        return psnr

    optimal_s = minimize(maximize_s, [1]).x[0]
    
    return psnr_hvs_m(ground_truth, optimal_s * reconstruction, CSFCof)[0]
  

def psnr_maximize(ground_truth, reconstruction):
    """Function takes in as parameters ground truth image and the reconstructed image. Images are not necessarily of the same scale.
    Optimization of the scaling parameter is performed within the function.
    
    Returns maximized PSNR"""
    d_range = ground_truth.max() - ground_truth.min()

    def maximize_s(s):
        psnr = compare_psnr(ground_truth, s * reconstruction, data_range=d_range) * -1
        return psnr

    optimal_s = minimize(maximize_s, [1]).x[0]
    
    return compare_psnr(ground_truth, optimal_s * reconstruction, data_range=d_range)


def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    PIXEL_MAX = img1.max()
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def fourier_freq_distance_MSE_1(x): #without tapering
    im_recon = np.array(x)
    gauss_kernel = np.outer(signal.gaussian(im_recon.shape[0], 3),
                        signal.gaussian(im_recon.shape[1], 3))

    freq_recon = fp.fft2(im_recon)

    freq_kernel_recon = fp.fft2(fp.ifftshift(gauss_kernel))
    convolved_recon = freq_recon*freq_kernel_recon
    im_output_recon = fp.ifft2(convolved_recon).real

    return im_output_recon


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    
    if window_len<3:
        return x

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def gamma(intensities, gamma_par=0.4):
    max_ = intensities.max()
    intensities_norm = intensities / max_
    transformed = intensities_norm**(1/gamma_par)
    transformed = transformed * max_
    return transformed

def gamma_inverse(intensities, gamma_par=0.4):
    return gamma(intensities, gamma_par=1./gamma_par)

def mixing_with_img(result: np.ndarray, reference_mask: np.ndarray, mask: np.ndarray, mixing_factor: float=1.5, hue: float=0.28, saturation: float=1) -> np.ndarray:
    from skimage.color import gray2rgb

    cum_image = gray2rgb(0.6 * reference_mask.copy()[:, :])
    image = result * mask / np.max(result * mask)
    image = gray2rgb(image * reference_mask[:, :])
    image = tomography.utils.colorize(image, hue=hue, saturation=saturation)  # green
    cum_image = np.maximum(mixing_factor * image, cum_image)
    return np.clip(cum_image, 0, 1)


# The following functions are required to create data for simulations. 
# The function create_filename_data will use the data and section it as would be done
# for creating strips of tissue (the secondary slices)

def get_angle(angle_list):
    """angle_list: list containing designated angles at positions 5:
    anglexxx
    for example: ['angle89', 'angle100']"""
    new_array = np.empty(0)
    array = [np.append(new_array, int(x[5:])) for x in angle_list]
    return np.array(array).flatten()


def random_2_points(dist, accu=10, xc= 100, yc=100):
    ang_dir = np.random.uniform(0, 2*np.pi)
    deltay = accu * np.sin(ang_dir)
    deltax = accu * np.cos(ang_dir)
    mu1 = np.array([xc + deltax, yc + deltay ])

    ang_dir = np.random.uniform(0, 2*np.pi)
    deltay = dist * np.sin(ang_dir)
    deltax = dist * np.cos(ang_dir)
    mu2 = mu1 + np.array([deltax, deltay])
    return mu1, mu2


def generate_gaussians(mu1, mu2, sigma1, sigma2, level_express=1, shape=(200,200)):
    X,Y = np.meshgrid(range(shape[0]), range(shape[1]))
    pos = np.column_stack([X.flat[:], Y.flat[:]])
    density = multivariate_normal.pdf(pos, mean=mu1, cov=sigma1) + multivariate_normal.pdf(pos, mean=mu2, cov=sigma2)
    average_expression = level_express * density / np.max(density)
    return (average_expression).reshape(shape)

def create_filename_data(file_path, angles_names, angles_values, var, factor, mask, width, error=True):
    """Creates h5py file with data. Noise is added automatically unless argument error=False
    
    Args:
    
    file_path: string path to .h5py
    angles_names: list of angle names in format anglexxx
    angles_values: integer values of angle names
    var: dictionary containing gene_list and attributes
    factor: amount of noise (poisson) to add
    error: if True, adds error, else data is taken as given"""
    
    # if os.path.isfile(file_path):
    #     !rm $file_path
    
    fout = h5py.File(file_path, 'w')
    D, proj_len = tomography.core.build_Design_Matrix(angles_values,
                                                  [width,]*len(angles_names),
                                                  mask,
                                                  0.1,
                                                  notation_inverse=True,
                                                  return_projlen=True)
    profiles = []
    for g_name, g_ in var.items():
        b_ = D.dot((g_.data*mask).flat[:])
        cum_proj_len = np.hstack([[0], proj_len]).cumsum()
        for i, name in enumerate(angles_names):
            if error:
                tmp = np.random.poisson(b_[cum_proj_len[i]:cum_proj_len[i+1]] / 
                                        np.mean(b_[cum_proj_len[i]:cum_proj_len[i+1]]) * factor, size=None)
                
                
            else:
                tmp = b_[cum_proj_len[i]:cum_proj_len[i+1]]
            fout.create_dataset("genes/%s/%s" % (g_name, name), data=tmp)
            profiles.append(tmp)
    for i, name in enumerate(angles_names):
        fout.create_dataset("coordinates/%s" % name, data=np.arange(proj_len[i]))
    
    fout.close()
    return profiles

