# author: baiCai

from torchvision import transforms as T
from torchvision.datasets import CIFAR10

# 这个简单，MNIST数据集很经典，因此可以直接调用官方的方法
def My_Dataset(train=True):
    # 定义预处理方法
    transforms = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    if train:
        #  路径需要修改为自己的
        # 另外，这个数据集会自动下载
        data = CIFAR10('../data/CIFAR10',train=True,download=True,transform=transforms)
    else:
        data = CIFAR10('../data/CIFAR10', train=False, download=True, transform=transforms)
    return data

