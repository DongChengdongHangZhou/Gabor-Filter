import numpy as np
import cv2
import random

def addNoise(image, block_noie_degree = 0.8):
    '''
    add simulated noise to the input skeleton image
    :param image: input image (skeleton)
    :param block_noie_degree: a float number ranges in [0,1] indicating the degree of noise,
                              0 means no noise, 1 means all noise.
    :return: the image with noise
    '''
    # adding block noise
    noise = np.random.randn(image.shape[0], image.shape[1])
    kernel1 = cv2.getGaussianKernel(11, 3)
    kernel2 = cv2.getGaussianKernel(31, 11)
    noise = cv2.sepFilter2D(noise, -1, kernel1, kernel1)
    noise = cv2.sepFilter2D(noise, -1, kernel2, kernel2)
    thr = noise.max() * (1-block_noie_degree)
    noise_map = noise > thr
    image_noise = image.copy()
    image_noise[noise_map] = 255

    # adding line structure noise
    #
    ind = np.nonzero(image==0)
    n = len(ind[0])
    R = np.random.permutation(range(n))
    for k in range(0,60,2):
        cv2.line(image_noise, (ind[1][R[k]],ind[0][R[k]]), (ind[1][R[k+1]],ind[0][R[k+1]]), (0,0,0))

    return image_noise

def segmentSkeleton(image):
    '''
    segment skeleton region
    :param image: the input skeleton image
    :return: the mask of skeleton region with 1 for foreground ans 0 for background
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21))
    mask = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    #cv2.imshow('mask', mask)
    #cv2.waitKey()
    return mask==0