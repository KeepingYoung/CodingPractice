import torch
from torch import nn
import numpy as np

class Generator(nn.Module):
    def __init__(self,in_dim,num_classes):
        '''
        :param in_dim: 噪声的维度
        :param num_classes: 类别个数，10个类别
        '''
        super().__init__()
        # 初始化
        self.in_dim = in_dim
        self.num_classes = num_classes

        # 定义模型
        self.model = nn.Sequential(
            # ConvTranspose2d：反卷积操作，将低维的噪声向量进行扩张到高纬度（生成图片的维度）
            # 关于ConvTranspose2d：https://blog.csdn.net/zhaohongfei_358/article/details/125639916
            # 网络输出：batch_size * 3 * 128 * 128
            nn.ConvTranspose2d(self.in_dim + self.num_classes, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64 * 4, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64 * 2, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64 * 2, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input_x):
        return self.model(input_x)
