import cv2
import numpy as np
from extraction.Fp import getOriGrad, enhance_fingerprint
from compute_freq import compute_global_freq


def drawOrientation(ori, background, mask=None, block_size=16, color=(255, 0, 0), thickness=1, is_block_ori=False):
    '''
    :param im: the background image to draw orientation field (in place, backup outside)
    :param ori: the pixel-wise orientation field
    :param blks_zie: the block size of orientation showing
    :param color: the line color showing the orientation
    :param thickness: the line thickness
    :return: im
    '''
    h = ori.shape[0]
    w = ori.shape[1]
    if is_block_ori:
        draw_step = 1
        offset = 0
    else:
        draw_step = block_size
        offset = int(block_size / 2)
    for x in range(0, w - offset, draw_step):
        for y in range(0, h - offset, draw_step):
            if mask is not None and mask[y + offset, x + offset] == 0:
                continue
            th = ori[y + offset, x + offset]
            if is_block_ori:
                x0 = x * block_size + block_size / 2
                y0 = y * block_size + block_size / 2
            else:
                x0 = x + block_size / 2
                y0 = y + block_size / 2
            x1 = int(x0 + 0.4 * block_size * np.cos(th))
            y1 = int(y0 + 0.4 * block_size * np.sin(th))
            x2 = int(x0 - 0.4 * block_size * np.cos(th))
            y2 = int(y0 - 0.4 * block_size * np.sin(th))
            cv2.line(background, (x1, y1), (x2, y2), color, thickness)
    return background

def showOrientation(ori, background=None, mask=None, block_size=16, color=(0,0,255), thickness=1,
                    win_name='orientation', wait_to_show=False):
    '''
    show orientation field in block wise. the orientation in each block is showing with a short line
    :param ori: the input orientation field in pixel-wise
    :param background: the background image to draw the lines. can be None, then show on a white image
    :param mask: the pixel-wise mask image
    :param block_size: the block size to draw orientation
    :param color: the color of line
    :param thickness: line thickness
    :param win_name: the window name
    :return: None
    '''
    if background is None:
        bg = np.ones((ori.shape[0], ori.shape[1], 3), dtype=np.uint8) * 255
    elif len(background.shape)==2:
        bg = np.stack((background, background, background), 2)
    else:
        bg = background.copy()

    im_show = drawOrientation(ori, bg, mask=mask, block_size=block_size, color=color, thickness=thickness)
    cv2.imshow(win_name, im_show)
    if not wait_to_show:
        cv2.waitKey()

if __name__ == '__main__':
    image = cv2.imread('./14.bmp', 0)
    # get_orientation
    ori = getOriGrad(image, w=31)
    #showOrientation(ori, image)
    h, w = image.shape
    h1, w1 = int(h / 2), int(w / 2)
    # get frequency
    f = 1. / compute_global_freq(image[h1 - 41: h1 + 41, w1 - 41: w1 + 41])
    image_enhance = enhance_fingerprint(image, ori, f=f, band_width=4.5) 
    #cv2.imshow('image', image)
    #cv2.imshow('image_enhance', image_enhance)
    cv2.imwrite('./save.bmp', image_enhance)
    #cv2.waitKey()
