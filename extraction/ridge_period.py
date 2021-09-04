import numpy as np
import cv2

def computeLocalRidgePeriod(im, ori, pt, block_width=16, block_height=64, min_t=4, max_t=16):
    '''
    compute local ridge period using the X-Signature method
    :param im: the gray-scale fingerprint image
    :param ori: orientation field in pixel-wise range in [0, pi)
    :param pt: the (x,y) coordinate of block center.
    :param block_width: the width of block window
    :param block_height: the height of block window
    :param min_t: the minimum value of ridge period (at 500 ppi resolution)
    :param max_t: the maximum value of ridge period (at 500 ppi resolution)
    :return: the ridge period value
    '''
    x0, y0 = pt
    x, y=np.meshgrid(range(-block_width/2,block_width/2),range(-block_height/2,block_height/2))
    u = (x * np.cos(ori[y0,x0]) + y * np.sin(ori[y0,x0]) + x0).astype(int)
    v = (x * np.sin(ori[y0,x0]) - y * np.cos(ori[y0,x0]) + y0).astype(int)
    u = u.reshape((block_width*block_height,))
    v = v.reshape((block_width*block_height,))
    L = (u>=0) & (v>=0) & (u<im.shape[1]) & (v<im.shape[0])
    msk = np.zeros(u.shape)
    msk[L] = 1
    sig = np.zeros(u.shape)
    sig[L] = im[v[L],u[L]]
    sig[L==0] = np.mean(sig[L])
    sig = sig.reshape((block_height,block_width))
    weight = msk.reshape(sig.shape).sum(1)
    #L = weight>0
    sig = np.sum(sig, 1)
    sig -= sig.mean()

    #plot(sig)
    #show()
    thr = np.ones(sig.shape)*(sig.max()+sig.min())/2.
    sig = sig>thr
    p = [i for i in range(1,len(sig)) if i>0 and abs(sig[i]-sig[i-1])!=0]
    T = [p[i]-p[i-1] for i in range(1,len(p))]
    if len(T)==0:
        return 9.

    T = np.mean(T) * 2
    T = min_t if T<min_t else T
    T = max_t if T>max_t else T
    return T

def getRidgePeriod(im, ori, step_size=16, block_width=16, block_height=64):
    '''
    get the ridge period field at block wise, and then interpolate to pixel-wise by nearest interpolation
    :param im: the gray-scale fingerprint image
    :param ori: orientation field in pixel-wise form
    :param step_size: the step size in computing the ridge period
    :param block_width: the width for X-signature window
    :param block_height: the height for X-signature window
    :return: the pixel wise ridge period
    '''
    blk_h = im.shape[0] / step_size
    blk_w = im.shape[1] / step_size
    ridge_period = np.zeros((blk_h, blk_w))
    for i in range(blk_h):
        for j in range(blk_w):
            ridge_period[i,j] = computeLocalRidgePeriod(im, ori, (j*step_size+step_size/2, i*step_size+step_size/2), block_width, block_height)
    ridge_period = cv2.resize(ridge_period, (im.shape[1], im.shape[0]), None, 0, 0, cv2.INTER_LINEAR)
    return ridge_period

def getSkeletonRidgePeriod(skeleton, ori, step_size=16, block_width=16, block_height=64):
    kernel = cv2.getGaussianKernel(9, 1.5)
    im = cv2.sepFilter2D(skeleton, -1, kernel, kernel)
    #cv2.imshow('im', im)
    #cv2.waitKey()
    period = getRidgePeriod(im, ori, step_size=step_size, block_width=block_width, block_height=block_height)
    return period