import numpy as np
from skimage.util.shape import view_as_windows


def memory_strided_im2col(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
        Perform the im2col computation on a 2D array arr
        Fix stride == 1; arr is meant to be Sypanse.ss
        Credit: https://towardsdatascience.com/how-are-convolutions-actually-performed-under-the-hood-226523ce7fbf
    """
    assert (arr.ndim == 2) and (arr.shape[0] == arr.shape[1]), "Require square 2D arr"
    assert (kernel.ndim == 2) and (kernel.shape[0] == kernel.shape[1]), "Require square 2D mask"
    output_shape = arr.shape[0] - kernel.shape[0] + 1
    return view_as_windows(arr, kernel.shape).reshape(output_shape*output_shape,
                                                      kernel.shape[0]**2)

def convolute(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """ Returns 1D array; reshape outside of function """
    mem_strided_arr = memory_strided_im2col(arr, kernel)
    return np.dot(mem_strided_arr, kernel.flatten())