import numpy as np 
import scipy as sp 
import cv2
from skimage.morphology import disk, binary_dilation, binary_erosion


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