#coding: utf-8
import numpy as np

def compute_global_freq(im, N=1024):
    im_fft_abs = np.abs(np.fft.fft2(im - im.mean(), (N, N)))
    im_fft_abs = np.fft.fftshift(im_fft_abs)

    x = np.linspace(0, N, N, endpoint=False) - N / 2.
    y = np.linspace(0, N, N, endpoint=False) - N / 2.
    xv, yv = np.meshgrid(x, y)
    r = np.sqrt(xv ** 2 + yv ** 2)

    hist, bins = np.histogram(r, np.linspace(0, N/2, int(N/2),endpoint=False), (0, N / 2),
                              weights=im_fft_abs)
    i = np.argmax(hist[88:162])+88
    k = bins[i]

    # 返回脊线频率值（即脊线宽度的倒数）。
    return float(k) / N
