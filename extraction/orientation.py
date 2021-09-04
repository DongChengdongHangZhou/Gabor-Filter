import cv2
import numpy as np

def getOriGrad(im, w=15):
    '''
    gradient based algorithm (i.e. Asker's algorithm) for fingerprint orientation field extraction
    :param im: gray-scale fingerprint image from which to extract the orientation field
    :param w: the window size to average the dxx, dyy and dxy, larger window size result in smoother orientation field
    :return: pixel wise orientation field image, range in [0, pi)
    '''
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(im.astype(float), cv2.CV_64FC1, 1, 0, None, 3)
    dy = cv2.Sobel(im.astype(float), cv2.CV_64FC1, 0, 1, None, 3)
    dxx = dx * dx
    dyy = dy * dy
    dxy = dx * dy
    dxx = cv2.blur(dxx, (w, w))
    dyy = cv2.blur(dyy, (w, w))
    dxy = cv2.blur(dxy, (w, w))
    ori = np.arctan2(2. * dxy, dxx - dyy) / 2 + np.pi / 2
    return ori

def getSkeletonOri(skeleton):
    '''
    compute the orientation field for skeleton image
    :param skeleton: input skeleton image
    :return: pixel-wise orientation field
    '''
    kernel = cv2.getGaussianKernel(9, 1.5)
    im = cv2.sepFilter2D(skeleton, -1, kernel, kernel)
    ori = getOriGrad(im, w=31)
    return ori

