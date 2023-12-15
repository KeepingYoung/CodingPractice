import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

from My_Dataset import My_Dataset
from network_files.Discriminator import Discriminator as D
from network_files.Generator import Generator as G

import numpy as np
import torch
from torch import nn
import torchvision.utils
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import optim
from torchvision.utils import save_image
from sklearn.preprocessing import LabelBinarizer

# 对生成器和判别器的网络参数做初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


#这里是初始化，意思是这是一个关于0到9总共10类的分类器
#每个类使用独热编码
lb = LabelBinarizer()
lb.fit(list(range(0, 10)))

# 将标签进行one-hot编码
def to_categrical(y: torch.FloatTensor):
    y_one_hot = lb.transform(y.cpu())
    floatTensor = torch.FloatTensor(y_one_hot)
    return floatTensor.to(device)

# 样本和one-hot标签进行连接
# data：(batch_size, 3, 128, 128)
# y: (batch_size, 1) 经过独热编码之后为 (batch_size, 10)
def concanate_data_label(data, y):
    y_one_hot = to_categrical(y)
    return torch.cat((data, y_one_hot), 1)

# 定义基本参数
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 设备
batch_size = 256
num_classes = 10
in_dim = 100  # 噪声维度

# 创建模型并对参数进行初始化
Generator = G(in_dim, num_classes).cuda()
Discriminator = D(num_classes).cuda()
Generator.model.apply(weights_init)
Discriminator.model.apply(weights_init)
Generator.to(device)
Discriminator.to(device)

# 定义损失函数：多标签分类任务
loss = nn.BCELoss()

# 优化器
optim_G = torch.optim.Adam(Generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optim_D = torch.optim.Adam(Discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 加载数据
train_dataset = My_Dataset(train=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           num_workers=8, pin_memory=True)

# 开始训练
# 固定生成器，训练判别器
for epoch in range(500):
    print('epoch: %4d' % (epoch))
    for batch, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        # =========================================================================
        # 拼接真实数据和标签
        # unsqueeze()：起升维的作用,参数表示在哪个地方加一个维度
        # to_categrical(target)：(batch_size, 10)
        # target1: (batch_size, 10, 1, 1)
        # repeat(): 参数是对应维度的复制个数
        # target2: (batch_size, 10, 128, 128)
        target1 = to_categrical(target).unsqueeze(2).unsqueeze(3).float()  # 加到噪声上
        target2 = target1.repeat(1, 1, data.size(2), data.size(3))  # 加到数据上
        # 将标签与数据拼接
        # (batch_size, 3, 128, 128)+(batch_size, 10, 128, 128) = (batch_size,13,128,128)
        data = torch.cat((data, target2), dim=1)
        # 真实数据标签为1
        label = torch.full((data.size(0), 1), 1.0).to(device)

        # 把真实数据+标签喂到判别器中
        Discriminator.zero_grad()
        output = Discriminator.forward(data)
        loss_D1 = loss(output, label)
        loss_D1.backward()

        # =========================================================================
        # 拼接噪声和标签
        # noise_z：(batch_size, in_dim, 1, 1)
        # target1: (batch_size, 10, 1, 1)
        # noise_z：(batch_size, in_dim + 10, 1, 1)
        noise_z = torch.randn(data.size(0), in_dim, 1, 1).to(device)
        noise_z = torch.cat((noise_z, target1), dim=1)  # 喂到生成器中的数据

        # 将noise_z喂到生成器中生成假图片：(batch_size, 3, 128, 128)
        # 和data一样，fake_data也要和标签向量target2连接起来
        fake_data = Generator.forward(noise_z)
        fake_data = torch.cat((fake_data, target2), dim=1)
        # 此时是生成的照片，标签为0
        label = torch.full((data.size(0), 1), 0.0).to(device)

        # 把假数据+标签喂到判别器中
        output = Discriminator.forward(fake_data.detach())
        loss_D2 = loss(output, label)
        loss_D2.backward()

        # =========================================================================
        # 更新判别器
        optim_D.step()

        # =========================================================================
        # =========================================================================
        # 计算生成图片和真实标签的损失，以此来更新生成器的参数
        # 因为生成器需要更可能地生成真实图片
        Generator.zero_grad()
        label = torch.full((data.size(0), 1), 1.0).to(device)
        output = Discriminator.forward(fake_data.to(device))
        lossG = loss(output, label)
        lossG.backward()

        # 更新生成器参数
        optim_G.step()

        # 每5个epoch保存图片，记录训练过程
        if epoch % 5 == 0 and batch == 0:
            # 生成指定target_label的图片
            noise_z1 = torch.randn(data.size(0), in_dim, 1, 1).to(device)
            # torch.full((data.size(0), 1), 4)模拟从迭代器中抽取的target
            # target是(batch_size, 1)的张量，这里设置为4，之后转化为独热编码，维度拓展之后和噪声连接
            target3 = to_categrical(torch.full((data.size(0), 1), 4)).unsqueeze(2).unsqueeze(3).float()  # 加到噪声上
            noise_z = torch.cat((noise_z1, target3), dim=1)  # (N,nz+n_classes,1,1)
            # 生成假图片
            fake_data = Generator.forward(noise_z.to(device))
            save_image(fake_data[:16] * 0.5 + 0.5,
                       'fake_images/epoch_%d_grid.png' % (epoch),
                       nrow=4,
                       normalize=True)

    if epoch % 10 == 0:
        # 保存模型
        state = {
            'net_G': Generator.state_dict(),
            'net_D': Discriminator.state_dict(),
            'start_epoch': epoch + 1
        }
        torch.save(state, './checkpoint/GAN_best_%d.pth' % (epoch))



