# author: baiCai

import torch
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image

from network_files.Generator import Generator as G
from sklearn.preprocessing import LabelBinarizer


#这里是初始化，意思是这是一个关于0到9总共10类的分类器
#每个类使用独热编码
lb = LabelBinarizer()
lb.fit(list(range(0, 10)))

# 将标签进行one-hot编码
def to_categrical(y: torch.FloatTensor):
    y_one_hot = lb.transform(y)
    floatTensor = torch.FloatTensor(y_one_hot)
    return floatTensor

num_classes = 10
in_dim = 100
batch_size = 256
# 创建生成器模型
Generator = G(in_dim, num_classes)

# 加载参数
Generator.load_state_dict(torch.load('./checkpoint/GAN_best_%d.pth'%(500))['net_G'])

for i in range(10):
    # 生成指定target_label的图片
    noise_z1 = torch.randn(batch_size, in_dim, 1, 1)
    # torch.full((data.size(0), 1), 4)模拟从迭代器中抽取的target
    # target是(batch_size, 1)的张量，这里设置为4，之后转化为独热编码，维度拓展之后和噪声连接
    target3 = to_categrical(torch.full((batch_size, 1), i)).unsqueeze(2).unsqueeze(3).float()
    noise_z = torch.cat((noise_z1, target3), dim=1)  # (N,nz+n_classes,1,1)
    # 生成假图片
    fake_data = Generator.forward(noise_z)
    save_image(fake_data[:16] * 0.5 + 0.5,
               'generate_images/class%d/gen_class%d_imgs.png' % (i+1,i+1),
               nrow=4,
               normalize=True)