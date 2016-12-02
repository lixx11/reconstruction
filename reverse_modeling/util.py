import numpy as np 
import scipy as sp 
import cv2
import h5py
import datetime
from skimage.morphology import disk, binary_dilation, binary_erosion
import math


def calc_custom_weight(image_shape, center, mode='exp', param=None):
    """Calculate the custom weight according to the mode and param.
    
    Args:
        image_shape (array_like): Image shape. 
        center (array_like): Center of image center.
        mode (str, optional): Weight mode. Default is the 'exp', 1 / exp(-r / k)
        param (None, optional): Parameter(s) to calculate weight, varies with different mode.
    
    Returns:
        TYPE: Weight matrix/image. The weight is calculated according to the mode and param.
    """
    center = np.asarray(center, dtype=np.float64)
    assert center.size == 2
    image_shape = np.asarray(image_shape, dtype=np.int)
    assert image_shape.size == 2
    y, x = np.indices(image_shape)
    r = np.sqrt((x - center[0])**2. + (y - center[1])**2.).astype(np.int)
    weight = np.zeros(image_shape)
    if mode == 'exp':
        param = np.asarray(param, dtype=np.float64)
        assert param.size == 1
        k = float(param)
        for radii in xrange(r.max()):
            weight[r==radii] = math.exp(-radii/k)
        return weight
    else:
        return None

def calc_SAXS_weight(image, center, mask=None, ignore_negative=True):
    """Summary
    
    Args:
        image (2d array): Input diffraction image.
        center (array_like): Center of input image.
    
    Returns:
        TYPE: Weight matrix/image. The weight is calculated by 1 / SAXS(r).
    """
    image = np.asarray(image, dtype=np.float64)
    assert len(image.shape) == 2
    center = np.asarray(center, dtype=np.float64)
    assert center.size == 2
    if mask is not None:
        mask = np.asarray(mask, dtype=np.float64)
        assert mask.shape == image.shape
        assert mask.min() >= 0. and mask.max() <= 1.
        mask = (mask > 0.5).astype(np.float64)
    else:
        mask = np.ones_like(image)
    if ignore_negative:
        mask *= (image > 0)
    SAXS = calc_radial_profile(image, center, mask=mask, mode='mean')
    y, x = np.indices((image.shape))
    r = np.round(np.sqrt((x - center[0])**2. + (y - center[1])**2.)).astype(np.int)
    weight = np.zeros_like(image)
    for radii in xrange(int(min(center[0], center[1], image.shape[1]-center[0], image.shape[0]-center[1]))):
        weight[r==radii] = 1./SAXS[radii]
    weight[np.isinf(weight)] = 0
    return weight


def calc_radial_profile(image, center, binsize=1., mask=None, mode='sum'):
    """Summary
    
    Parameters
    ----------
    image : 2d array
        Input image to calculate radial profile
    center : array_like with 2 elements
        Center of input image
    binsize : float, optional
        By default, the binsize is 1 in pixel.
    mask : 2d array, optional
        Binary 2d array used in radial profile calculation. The shape must be same with image. 1 means valid while 0 not.
    mode : {'sum', 'mean'}, optional
        'sum'
        By default, mode is 'sum'. This returns the summation of each ring.
    
        'mean'
        Mode 'mean' returns the average value of each ring.
    
    Returns
    -------
    Radial profile: 1d array
        Output array, contains summation or mean value of each ring with binsize of 1 along rho axis.
    
    Raises
    ------
    ValueError
        Description
    """
    image = np.asarray(image, dtype=np.float64)
    assert len(image.shape) == 2
    center = np.asarray(center, dtype=np.float64)
    assert center.size == 2
    if mask is not None:
        mask = np.asarray(mask, dtype=np.float64)
        assert mask.shape == image.shape
        assert mask.min() >= 0. and mask.max() <= 1.
        mask = (mask > 0.5).astype(np.float64)
    else:
        mask = np.ones_like(image)
    y, x = np.indices((image.shape))
    r = np.sqrt((x - center[0])**2. + (y - center[1])**2.)
    bin_r = r / binsize
    bin_r = np.round(bin_r).astype(int)
    radial_sum = np.bincount(bin_r.ravel(), image.ravel())  # summation of each ring

    if mode == 'sum':
        return radial_sum
    elif mode == 'mean':
        if mask is None:
            mask = np.ones(image.shape)
        nr = np.bincount(bin_r.ravel(), mask.ravel())
        radial_mean = radial_sum / nr
        radial_mean[np.isinf(radial_mean)] = 0.
        radial_mean[np.isnan(radial_mean)] = 0.
        return radial_mean
    else:
        raise ValueError('Wrong mode: %s' %mode)


def dict2h5(dict_data, output):
    """Write dict data to given output in h5 format
    
    Args:
        dict_data (dict): Description
        output (str): Output filepath
    
    Returns:
        TYPE: None
    """
    h5_file = h5py.File(output, 'w')
    for key in dict_data:
        h5_file.create_dataset(key, data=np.asarray(dict_data[key])) 
    h5_file.close()


def imrotate(img, center=None, angle=0.):
    """Summary
    
    Args:
        img (ndarray): Input image to rotate
        center (array_like): Rotation center.
        angle (float): Rotation angle in degree.
    
    Returns:
        TYPE: ndarray with the same size of img
    """
    img = np.asarray(img, dtype=np.float)
    rows, cols = img.shape
    if center is None:
        center = np.array([rows/2., cols/2.], dtype=np.float)
    else:
        center = np.asarray(center, dtype=np.float)
    assert center.size == 2
    M = cv2.getRotationMatrix2D((center[0], center[1]), angle, 1)
    rot_img = cv2.warpAffine(img, M, (cols,rows))
    return rot_img


def load_model(filepath, model_size, space_size):
    """load model from .npy file, return space with model
    
    Args:
        filepath (TYPE): Description
        model_size (TYPE): Description
        space_size (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    image = np.load(filepath)
    image = cv2.resize(image, (model_size, model_size))
    space_center = (space_size - 1) // 2
    model_center = (model_size - 1) // 2
    model_range = (space_center - model_center, space_center - model_center + model_size)
    space = np.zeros((space_size, space_size), dtype=np.float)
    space[model_range[0]:model_range[1], model_range[0]:model_range[1]] = image
    return space


def make_model(model_size, space_size):
    """make square model, return space with model
    
    Args:
        model_size (TYPE): Description
        space_size (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    space = np.zeros((space_size, space_size), dtype=np.float)
    space_center = (space_size - 1) // 2
    model_center = (model_size - 1) // 2
    model_range = (space_center - model_center, space_center - model_center + model_size)
    space[model_range[0]:model_range[1], model_range[0]:model_range[1]] = 1.
    return space


def get_edge(image, width=1, find_edge='both'):
    """Return edge points of given image, including outer and inner edge with specified width.
    
    Args:
        image (ndarray): Input image, 0 is backgroud.
        width (int, optional): Width of edge. Default is 1.
        find_edge ({'inner', 'outer', 'both'}, optional): Edge type. Default is both.
    
    Returns:
        ndarray: binary image where the edge point is True.
    
    """
    selem = disk(width)
    input_img = image > 0
    erosion_img = binary_erosion(image, selem)
    dilation_img = binary_dilation(image, selem)
    if find_edge == 'both':
        edge = np.logical_xor(dilation_img, erosion_img)
    elif find_edge == 'outer':
        edge = np.logical_xor(dilation_img, input_img)
    elif find_edge == 'inner':
        edge = np.logical_xor(input_img, erosion_img)
    else:
        print('ERROR! Unrecognized edge-finding method: %s' %find_edge)            
    return edge


def make_square_mask(image_size, mask_size):
    """Summary
    
    Args:
        image_size (int): Size of image in each dimension of square. Odd number is better.
        mask_size (int): Size of mask in each dimension of square. Odd number is better.
    
    Returns:
        ndarray: 2d image with specified-size mask
    """
    img = np.ones((image_size, image_size))
    img_hs = (image_size - 1) // 2 # half size of image
    mask_hs = (mask_size - 1) // 2 # half size of mask
    mask_range = (img_hs - mask_hs, img_hs - mask_hs + mask_size - 1)
    img[mask_range[0]:mask_range[1]+1, mask_range[0]:mask_range[1]+1] = 0.
    return img


def get_mass_center(image):
    """Calculate mass center of give image.
    
    Args:
        image (ndarray): Input 2d image. 0 is backgroud.
    
    Returns:
        (x, y): Center of Mass of input image
    """
    ys, xs = np.where(image)
    center_x = int(np.round(xs.mean()))
    center_y = int(np.round(ys.mean()))
    return center_y, center_x