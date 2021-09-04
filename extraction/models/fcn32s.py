import numpy as np
import torch
import torch.nn as nn

class FCN32(nn.Module):

    def __init__(self):
        super(FCN32, self).__init__()
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(1, 64, 3, padding=100),
	    nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
	    nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2

            # conv2
            nn.Conv2d(64, 128, 3, padding=1),
	    nn.BatchNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
	    nn.BatchNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4

            # conv3
            nn.Conv2d(128, 256, 3, padding=1),
	    nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
	    nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
	    nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8

            # conv4
            nn.Conv2d(256, 512, 3, padding=1),
	    nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
	    nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
	    nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16

            # conv5
            nn.Conv2d(512, 512, 3, padding=1),
	    nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
	    nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
	    nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/32
        )
        self.ori_out = nn.Sequential(
            # fc6
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            # fc7
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            # ori
            nn.Conv2d(4096, 2, 1),
            # make original size 
            nn.ConvTranspose2d(2, 2, 64, stride=32, bias=False),
        )
        self.rp_out = nn.Sequential(
            # fc6
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            # fc7
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            # ori
            nn.Conv2d(4096, 1, 1),
            # make original size 
            nn.ConvTranspose2d(1, 1, 64, stride=32, bias=False),
        )
        self.mask_out = nn.Sequential(
            # fc6
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            # fc7
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            # ori
            nn.Conv2d(4096, 2, 1),
            # make original size 
            nn.ConvTranspose2d(2, 2, 64, stride=32, bias=False),
        )

    def forward(self, x):
        h = self.features(x)
        ori = self.ori_out(h)
        rp = self.rp_out(h)
        mask = self.mask_out(h)
        ori = ori[:, :, 19:19+x.size()[2], 19:19+x.size()[3]].contiguous()
        rp = rp[:, :, 19:19+x.size()[2], 19:19+x.size()[3]].contiguous()
        mask = mask[:, :, 19:19+x.size()[2], 19:19+x.size()[3]].contiguous()
        return [ori, rp, mask]
