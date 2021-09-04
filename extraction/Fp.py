import cv2
import numpy as np
#from pylab import *
from .libfp import *
#from skimage import morphology
from .orientation import getOriGrad
from matplotlib import pylab

# def draw_point_list(im, point_list, radius=3, color=(255, 0, 0), thickness=1):
#     '''
#     :param im: the background image to draw the points (in place)
#     :param point_list: the point list with each element a tuple of coordinate in the form (x,y)
#     :param radius: the radius of circles
#     :param color: the color of circles
#     :param thickness: the thickness of line
#     :return: im
#     '''
#     [cv2.circle(im, (int(c[1]), int(c[0])), radius, color, thickness) for c in point_list]
#
#
#
# def get_singular_points(ori):
#     """
#     extract singular points from a given orientation field using the method
#     proposed in Alice(Yi) Wang's PAMI paper(FOMFE)
#     :param ori: the pixel-wise orientation field
#     :return: core, delta. core is a list of (x,y,1,dir) and delta is a list of (x, y, 0)
#     """
#     h,w=ori.shape
#     cos2 = np.cos(2*ori)
#     sin2 = np.sin(2*ori)
#     cos2_x = cv2.Sobel(cos2, cv2.CV_64FC1, 1, 0, None, 3)
#     cos2_y = cv2.Sobel(cos2, cv2.CV_64FC1, 0, 1, None, 3)
#     sin2_x = cv2.Sobel(sin2, cv2.CV_64FC1, 1, 0, None, 3)
#     sin2_y = cv2.Sobel(sin2, cv2.CV_64FC1, 0, 1, None, 3)
#     A = cos2_x * sin2_y - cos2_y*sin2_x
#     A = cv2.GaussianBlur(A, (3,3), 2)
#     #imshow(A,'gray')
#     #show()
#
#     y,x = np.where(np.abs(A)>10)
#     sp = [(i, j, 1 if A[j,i]>0 else 0) for i, j in zip(x, y) if
#           i>0 and j>0 and i<w-1 and j<h-1 and
#           abs(A[j,i])>abs(A[j+1,i]) and
#           abs(A[j,i])>abs(A[j,i+1]) and
#           abs(A[j,i])>abs(A[j-1,i]) and
#           abs(A[j,i])>abs(A[j,i-1])]
#
#     # get core direction
#     def get_core_dir(ori, s):
#         r = np.min(np.array((s[0]-1, s[1]-1, w-s[0]-2, h-s[1]-2, 50)))
#         o = [ori[int(s[1]+r*np.sin(t)), int(s[0]+r*np.cos(t))]
#              for t in np.arange(0, np.pi*2, np.pi*2/360.)]
#         d = [abs(cos(t1)*cos(t2)+sin(t1)*sin(t2)) for t1,t2 in zip(o,np.arange(0, np.pi*2, np.pi*2/360.))]
#         t = np.argmax(d)
#         return t * np.pi*2/360.
#
#     core = [(s[0], s[1], s[2], get_core_dir(ori, s)) for s in sp if s[2]==1]
#     delta = [(s[0], s[1], s[2]) for s in sp if s[2]==0]
#     return core, delta
#
# def get_gradient_map(im):
#     '''
#     extract the gradients of a image using sobel operator
#     :param im: the input image to extract gradients
#     :return: grad, dx, dy, where grad = |dx| + |dy|
#     '''
#     dx = cv2.Sobel(im.astype(float), cv2.CV_64FC1, 1, 0, None, 1)
#     dy = cv2.Sobel(im.astype(float), cv2.CV_64FC1, 0, 1, None, 1)
#     grad = np.abs(dx) + np.abs(dy)
#     return grad, dx, dy
#
def get_gabor_filter(length, sigma, freq, normalize=True):
    '''
    get a 1D gabor filter. g(x) = exp(-x^2/(2sigma^2)) * cos(2*x*freq), the filter will be normalized
    if the normalize parameter set
    :param length: length of Gabor filter, should be an odd number
    :param sigma: sigma of Gaussian
    :param freq: center frequency
    :param normalize: indicator whether to normalize the output filter
    :return: 1D Gabor filter
    '''
    x = np.arange(-(length / 2), length / 2 + 1, dtype=np.float64)
    g = np.exp(-(x * x) / (2. * sigma * sigma)) * np.cos(2 * np.pi * freq * x)
    if normalize:
        g = g / np.linalg.norm(g)
    return g
#
# def get_gabor_filter_list(length, sigma, min_period, max_period, period_step=0.5):
#     '''
#     get a list of gabor filter.
#     :param length: length of Gabor filter, should be an odd number
#     :param sigma: sigma of Gaussian
#     :param min_period: minimum  ridge period
#     :param max_period: maximum  ridge period
#     :return: a set of Gabor filters with each column a Gabor filter
#     '''
#     G = [get_gabor_filter(length, sigma, 1./T) for T in np.arange(min_period, max_period, step=period_step)]
#     G = np.row_stack(G)
#     return G
#
def get_gaussian_filter(length, sigma, normalize=True):
    '''
    get a Gaussian filter: g = np.exp(-(x * x) / (2. * sigma * sigma))
    :param length: the length of gaussian filter
    :param sigma: sigma of gaussian
    :return: a 1D gaussian filter
    '''
    x = np.arange(-(length / 2), length / 2 + 1, dtype=np.float64)
    g = np.exp(-(x * x) / (2. * sigma * sigma))
    if normalize:
        g = g / np.linalg.norm(g)
    return g
#
# def gaussian_derivative(winsize, sigma):
#     gaussian = np.zeros((winsize,winsize), dtype=np.complex)
#     for x in range(-(winsize/2), winsize/2+1):
#         for y in range(-(winsize/2), winsize/2+1):
#             dx = np.exp(-(x*x+y*y)/(2*sigma*sigma))*(-x/(sigma*sigma))/np.sqrt(2*np.pi)/sigma
#             dy = np.exp(-(x*x+y*y)/(2*sigma*sigma))*(-y/(sigma*sigma))/np.sqrt(2*np.pi)/sigma
#             gaussian[y+winsize/2, x+winsize/2] = dx + dy*1j
#     return gaussian
#
# def threshold_enh(enh):
#     grad = get_gradient_map(enh)[0]
#     avg = cv2.blur(grad, (11, 11))
#     skl = morphology.skeletonize(grad > avg)
#     thr = np.zeros(enh.shape, np.uint8)
#     thr[skl == 1] = enh[skl == 1]
#     fill_threshold(thr)
#     # cv2.blur(thr,(21,21), thr)
#     cv2.medianBlur(thr, 13, thr)
#     cv2.blur(thr, (5, 5), thr)
#     return thr
#
# def direct_grad(im, ori, block_width, block_height):
#     kernel = -1*np.ones((block_height,), float)
#     kernel[block_height/2] = block_height-1
#     grad = ridge_filter(im.astype(np.float64), ori, kernel)
#     h = get_gaussian_filter(block_width, 1)/block_width
#     grad = ridge_filter(grad, ori+np.pi/2, h)
#     return grad
#
#
# def remove_boundary_point(bin, x, y):
#     p = bin[np.ix_(y,x+1)] & bin[np.ix_(y,x-1)] & bin[np.ix_(y+1,x)] & bin[np.ix_(y-1,x)]
#     x = x[p]
#     y = y[p]
#     return x,y
#
# def find_object(bin):
#     pass
#
# def draw_singular_points(im, sp):
#     [cv2.circle(im, s[:2], 5, (255,0,0), 2) for s in sp]
#     [cv2.line(im, s[:2], (int(s[0]+20*cos(s[3])), int(s[1]+20*sin(s[3]))), (255,0,0), 2) for s in sp]

def enhance_fingerprint(im, ori, gaussian_len=9, gabor_len=21, f=10., band_width=2.):
    #ori = getOriGrad(im, w)
    gaussian = get_gaussian_filter(gaussian_len, 4)
    gabor = get_gabor_filter(gabor_len, band_width, 1./f)
    enh = ridge_filter(im, ori + np.pi / 2., gabor)
    enh = ridge_filter(enh, ori, gaussian)
    cv2.normalize(enh, enh, 0, 255, cv2.NORM_MINMAX)
    return enh.astype(np.uint8)
