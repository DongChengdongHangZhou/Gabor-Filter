from ctypes import *
import numpy as np
import os

# load shared library
p = os.path.abspath(__file__)
folder, title = os.path.split(p)
libfp = np.ctypeslib.load_library(os.path.join(folder, '_libfp.so'), '.')


# declare functions
libfp.ridge_filter.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
    c_long,
    c_long,
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    c_long
]

# declare functions
libfp.ridge_gabor_filter.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
    c_long,
    c_long,
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
    c_long,
    c_long,
    c_double,
    c_double
]




libfp.fill_threshold.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2, flags="C_CONTIGUOUS"),
    c_long,
    c_long
]

libfp.pore_map.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
    c_long,
    c_long,
    c_long,
    c_double
]

libfp.pore_map2.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
    c_long,
    c_long,
    c_long
]

libfp.find_peaks.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2, flags="C_CONTIGUOUS"),
    c_long,
    c_long,
    c_long
]

libfp.ridge_median_filter.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
    c_long,
    c_long,
    c_long
]


# wrap all functions
def ridge_filter(img, ori, kernel, out=None):
    if out is None:
        out = np.zeros(img.shape, dtype=np.float64)

    libfp.ridge_filter(img.astype(np.float64), ori.astype(np.float64), out, img.shape[1], img.shape[0], kernel, kernel.shape[0])
    return out

def ridge_gabor_filter(img, ori, period, gabor, min_period, max_period, out=None):
    if out is None:
        out = np.zeros(img.shape, dtype=np.float64)
    libfp.ridge_gabor_filter(img.astype(float), ori, period, out, img.shape[1], img.shape[0],
                             gabor, gabor.shape[1], gabor.shape[0], min_period, max_period)
    return out

def fill_threshold(thr):
    libfp.fill_threshold(thr, thr.shape[1], thr.shape[0])

def pore_map(img, R, sigma, out=None):
    if out is None:
        out = np.zeros(img.shape, dtype=np.float64)
    libfp.pore_map(img, out, img.shape[1], img.shape[0], int(R), float(sigma))
    return out

def pore_map2(img, R, out=None):
    if out is None:
        out = np.zeros(img.shape, dtype=np.float64)
    libfp.pore_map2(img, out, img.shape[1], img.shape[0], int(R))
    return out

def find_peaks(im, thr, msk=None):
    out = np.zeros(im.shape, np.uint8)
    libfp.find_peaks(im.astype(np.uint8), out, im.shape[1],im.shape[0],thr)
    if msk is None:
        return np.where(out)
    else:
        return np.where(out & msk)

def ridge_median_filter(im, ori, kernel_size, out=None):
    if out is None:
        out = np.zeros(im.shape, float)
    libfp.ridge_median_filter(im.astype(np.float64), ori, out, im.shape[1], im.shape[0], kernel_size)
    return out