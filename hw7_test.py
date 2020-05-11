import os
import torch
import time
import torchvision.transforms as transforms
from PIL import Image
from glob import glob
import re
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


def encode8(params, fname):
    custom_dict = {}
    for (name, param) in params.items():
        param = np.float64(param.cpu().numpy())
        if type(param) == np.ndarray:
            min_val = np.min(param)
            max_val = np.max(param)
            param = np.round((param - min_val) / (max_val - min_val) * 255)
            param = np.uint8(param)
            custom_dict[name] = (min_val, max_val, param)
        else:
            custom_dict[name] = param

    pickle.dump(custom_dict, open(fname, 'wb'))


def decode8(fname):
    params = pickle.load(open(fname, 'rb'))
    custom_dict = {}
    for (name, param) in params.items():
        if type(param) == tuple:
            min_val, max_val, param = param
            param = np.float64(param)
            param = (param / 255 * (max_val - min_val)) + min_val
            param = torch.tensor(param)
        else:
            param = torch.tensor(param)

        custom_dict[name] = param

    return custom_dict


def loss_fn_kd(outputs, labels, teacher_outputs, T=20, alpha=0.5):
    # 一般的Cross Entropy
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    # 讓logits的log_softmax對目標機率(teacher的logits/T後softmax)做KL Divergence。
    soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                                                    F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
    return hard_loss + soft_loss


"""# Data Processing

我們的Dataset使用的是跟Hw3 - CNN同樣的Dataset，因此這個區塊的Augmentation / Read Image大家參考或直接抄就好。

如果有不會的話可以回去看Hw3的colab。

需要注意的是如果要自己寫的話，Augment的方法最好使用我們的方法，避免輸入有差異導致Teacher Net預測不好。
"""


def readfile(folderName):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    print("Reading data from {}...".format(folderName))
    data = []
    label = []
    for img_path in sorted(glob(folderName + '/*.jpg')):
        try:
            # Get classIdx by parsing image path
            class_idx = int(re.findall(re.compile(r'\d+'), img_path)[1])
        except:
            # if inference mode (there's no answer), class_idx default 0
            class_idx = 0

        image = Image.open(img_path)
        # Get File Descriptor
        image_fp = image.fp
        image.load()
        # Close File Descriptor (or it'll reach OPEN_MAX)
        image_fp.close()

        data.append(image)
        label.append(class_idx)
    return data, label


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, x, y, transform=None):
        self.transform = transform
        self.data = x
        self.label = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.label[idx]


trainTransform = transforms.Compose([
    transforms.RandomCrop(256, pad_if_needed=True, padding_mode='symmetric'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])
testTransform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])


def get_dataset(x, y, mode='training', batch_size=32, trans=False):

    assert mode in ['training', 'testing', 'validation']

    dataset = MyDataset(
        x, y,
        transform=trainTransform if (trans) else testTransform)

    return dataset


def get_dataloader(dataset, mode='training', batch_size=32):

    assert mode in ['training', 'testing', 'validation']

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'training'))

    return dataloader


"""# Pre-processing

我們已經提供TeacherNet的state_dict，其架構是torchvision提供的ResNet18。

至於StudentNet的架構則在hw7_Architecture_Design.ipynb中。

這裡我們使用的Optimizer為AdamW，沒有為甚麼，就純粹我想用。
"""


class StudentNet(nn.Module):
    '''
      在這個Net裡面，我們會使用Depthwise & Pointwise Convolution Layer來疊model。
      你會發現，將原本的Convolution Layer換成Dw & Pw後，Accuracy通常不會降很多。

      另外，取名為StudentNet是因為這個Model等會要做Knowledge Distillation。
    '''

    def __init__(self, base=16, width_mult=1):
        '''
          Args:
            base: 這個model一開始的ch數量，每過一層都會*2，直到base*16為止。
            width_mult: 為了之後的Network Pruning使用，在base*8 chs的Layer上會 * width_mult代表剪枝後的ch數量。
        '''
        super(StudentNet, self).__init__()
        multiplier = [1, 2, 4, 8, 16, 16, 16, 16]

        # bandwidth: 每一層Layer所使用的ch數量
        bandwidth = [base * m for m in multiplier]

        # 我們只Pruning第三層以後的Layer
        for i in range(3, 7):
            bandwidth[i] = int(bandwidth[i] * width_mult)

        self.cnn = nn.Sequential(
            # 第一層我們通常不會拆解Convolution Layer。
            nn.Sequential(
                nn.Conv2d(3, bandwidth[0], 3, 1, 1),
                nn.BatchNorm2d(bandwidth[0]),
                nn.ReLU6(),
                nn.MaxPool2d(2, 2, 0),
            ),
            # 接下來每一個Sequential Block都一樣，所以我們只講一個Block
            nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(bandwidth[0], bandwidth[0],
                          3, 1, 1, groups=bandwidth[0]),
                # Batch Normalization
                nn.BatchNorm2d(bandwidth[0]),
                # ReLU6 是限制Neuron最小只會到0，最大只會到6。 MobileNet系列都是使用ReLU6。
                # 使用ReLU6的原因是因為如果數字太大，會不好壓到float16 / or further qunatization，因此才給個限制。
                nn.ReLU6(),
                # Pointwise Convolution
                nn.Conv2d(bandwidth[0], bandwidth[1], 1),
                # 過完Pointwise Convolution不需要再做ReLU，經驗上Pointwise + ReLU效果都會變差。
                nn.MaxPool2d(2, 2, 0),
                # 每過完一個Block就Down Sampling
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[1], bandwidth[1],
                          3, 1, 1, groups=bandwidth[1]),
                nn.BatchNorm2d(bandwidth[1]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[1], bandwidth[2], 1),
                nn.MaxPool2d(2, 2, 0),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[2], bandwidth[2],
                          3, 1, 1, groups=bandwidth[2]),
                nn.BatchNorm2d(bandwidth[2]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[2], bandwidth[3], 1),
                nn.MaxPool2d(2, 2, 0),
            ),

            # 到這邊為止因為圖片已經被Down Sample很多次了，所以就不做MaxPool
            nn.Sequential(
                nn.Conv2d(bandwidth[3], bandwidth[3],
                          3, 1, 1, groups=bandwidth[3]),
                nn.BatchNorm2d(bandwidth[3]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[3], bandwidth[4], 1),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[4], bandwidth[4],
                          3, 1, 1, groups=bandwidth[4]),
                nn.BatchNorm2d(bandwidth[4]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[5], bandwidth[5], 1),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[5], bandwidth[5],
                          3, 1, 1, groups=bandwidth[5]),
                nn.BatchNorm2d(bandwidth[5]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[6], bandwidth[6], 1),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[6], bandwidth[6],
                          3, 1, 1, groups=bandwidth[6]),
                nn.BatchNorm2d(bandwidth[6]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[6], bandwidth[7], 1),
            ),

            # 這邊我們採用Global Average Pooling。
            # 如果輸入圖片大小不一樣的話，就會因為Global Average Pooling壓成一樣的形狀，這樣子接下來做FC就不會對不起來。
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            # 這邊我們直接Project到11維輸出答案。
            nn.Linear(bandwidth[7], 11),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


teacher_net = models.resnet18(pretrained=False, num_classes=11).cuda()
student_net = StudentNet(base=16).cuda()
print("Total parameters:{}".format(sum(p.numel()
                                       for p in student_net.parameters())))
print("Trainable parameters:{}".format(sum(p.numel()
                                           for p in student_net.parameters() if p.requires_grad)))
print("{:=^30}".format("Start training!"))

teacher_net.load_state_dict(torch.load(f'./teacher_resnet18.bin'))
# 這裡記得要改
student_net.load_state_dict(decode8('./model.bin'))
optimizer = optim.SGD(student_net.parameters(), lr=1e-3, momentum=0.9)


''' load test data'''

test_x, test_y = readfile(os.path.join(sys.argv[1], 'testing'))


print('Size of Testing Data:{}'.format(len(test_x)))

# get dataloader

test_set = get_dataset(test_x, test_y, batch_size=32)
test_loader = get_dataloader(test_set, 'testing', batch_size=32)


''' predict'''
print("{:=^40}".format("Start Predicting..."))
student_net.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = student_net(data[0].cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)

# 將結果寫入 csv 檔
pred_file = sys.argv[2]
with open(pred_file, 'w') as f:
    f.write('Id,label\n')
    for i, y in enumerate(prediction):
        f.write('{},{}\n'.format(i, y))
