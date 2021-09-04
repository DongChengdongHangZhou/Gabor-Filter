import collections
import os.path as osp

import numpy as np
import PIL.Image
#import scipy.io
import torch
from torch.utils import data
import cv2


class V300ImageFeatures(data.Dataset):
    def __init__(self, root, name_list_file):
        self.root = root
	self.name_list_file = name_list_file
	self.files = [name.strip() for name in open(self.name_list_file).readlines()]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # load image
        img_file = osp.join(self.root, 'generated_with_noise', self.files[index])
	img = cv2.imread(img_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)

        # load ori
        ori_file = osp.join(self.root, 'features/orientation', self.files[index])
	ori_img = cv2.imread(ori_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)

        # load mask
        mask_file = osp.join(self.root, 'features/mask', self.files[index])
	mask_img = cv2.imread(mask_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)

        # load ridge period
        rp_file = osp.join(self.root, 'features/period', self.files[index])
	rp_img = cv2.imread(rp_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)

        return self.transform(img, ori_img, rp_img, mask_img)

    def transform(self, img, ori_img, rp_img, mask_img):
        # img
	img = img.astype(np.float64)
        img -= 128
	img /= 128
	img = np.expand_dims(img, 0)
        img = torch.from_numpy(img).float()
	
	# ori
	ori_img = ori_img.astype(np.float64)
	ori_img = ori_img * np.pi / 180.
	cos2ori = np.cos(2*ori_img)
	sin2ori = np.sin(2*ori_img)
	ori_norm = np.stack((cos2ori, sin2ori), axis=0)
        ori_norm = torch.from_numpy(ori_norm).float()
        
	# rp
	rp_img = rp_img.astype(np.float64) / 160.
	rp_img = np.expand_dims(rp_img, 0)
        rp_img = torch.from_numpy(rp_img).float()

	# mask
	mask_img[mask_img > 0] = 1
        mask_img = torch.from_numpy(mask_img).long()
	return img, ori_norm, rp_img, mask_img

    def untransform(self, img, ori_img, rp_img, mask_img):
	# img
        img = img.numpy()
        img = img * 128 + 128
        img = img.astype(np.uint8)
	# ori
        ori_img = ori_img.numpy()
	cos2ori = ori_img[0, :, :]
	sin2ori = ori_img[1, :, :]
	ori_img = np.arctan2(sin2ori, cos2ori) / 2. * 180. / np.pi
	ori_img[ori_img < 0] += 180
	ori_img = ori_img.astype(np.uint8)

	# rp
	rp_img = rp_img.numpy() * 160
	rp_img = rp_img.astype(np.uint8)

	# mask
	mask_img = mask_img.numpy() * 255
	mask_img = mask_img.astype(np.uint8)

        return img, ori_img, rp_img, mask_img
