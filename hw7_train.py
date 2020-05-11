# -*- coding: utf-8 -*-
# 用來加入validation set做訓練

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
"""Knowledge Distillation
===

<img src="https://i.imgur.com/H2aF7Rv.png=100x" width="500px">

簡單上來說就是讓已經做得很好的大model們去告訴小model"如何"學習。
而我們如何做到這件事情呢? 就是利用大model預測的logits給小model當作標準就可以了。

## 為甚麼這會work?
* 例如當data不是很乾淨的時候，對一般的model來說他是個noise，只會干擾學習。透過去學習其他大model預測的logits會比較好。
* label和label之間可能有關連，這可以引導小model去學習。例如數字8可能就和6,9,0有關係。
* 弱化已經學習不錯的target(?)，避免讓其gradient干擾其他還沒學好的task。


## 要怎麼實作?
* $Loss = \alpha T^2 \times KL(\frac{\text{Teacher's Logits}}{T} || \frac{\text{Student's Logits}}{T}) + (1-\alpha)(\text{原本的Loss})$


* 以下code為甚麼要對student使用log_softmax: https://github.com/peterliht/knowledge-distillation-pytorch/issues/2
* reference: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
"""


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

# get data
train_x, train_y = readfile(os.path.join(sys.argv[1], 'training'))
val_x, val_y = readfile(os.path.join(sys.argv[1], 'validation'))

train_x = train_x + val_x
train_y = train_y + val_y
print('Size of Training Data:{}'.format(len(train_x)))
print('Size of Validation Data:{}'.format(len(val_x)))
# get dataloader

train_set = get_dataset(train_x, train_y, batch_size=32, trans=True)
valid_set = get_dataset(val_x, val_y, batch_size=32)
train_dataloader = get_dataloader(train_set, 'training', batch_size=32)
valid_dataloader = get_dataloader(valid_set, 'validation', batch_size=32)


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
student_net.load_state_dict(torch.load('./model.bin'))
optimizer = optim.SGD(student_net.parameters(), lr=1e-4, momentum=0.9)

"""# Start Training

* 剩下的步驟與你在做Hw3 - CNN的時候一樣。

## 小提醒

* torch.no_grad是指接下來的運算或該tensor不需要算gradient。
* model.eval()與model.train()差在於Batchnorm要不要紀錄，以及要不要做Dropout。
"""


def run_epoch(dataloader, update=True, alpha=0.5):
    total_num, total_hit, total_loss = 0, 0, 0
    for now_step, batch_data in enumerate(dataloader):
        # 清空 optimizer
        optimizer.zero_grad()
        # 處理 input
        inputs, hard_labels = batch_data
        inputs = inputs.cuda()
        hard_labels = torch.LongTensor(hard_labels).cuda()
        # 因為Teacher沒有要backprop，所以我們使用torch.no_grad
        # 告訴torch不要暫存中間值(去做backprop)以浪費記憶體空間。
        with torch.no_grad():
            soft_labels = teacher_net(inputs)

        if update:
            logits = student_net(inputs)
            # 使用我們之前所寫的融合soft label&hard label的loss。
            # T=20是原始論文的參數設定。
            loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
            loss.backward()
            optimizer.step()

        else:
            # 只是算validation acc的話，就開no_grad節省空間。
            with torch.no_grad():
                logits = student_net(inputs)
                loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)

        total_hit += torch.sum(torch.argmax(logits, dim=1)
                               == hard_labels).item()
        total_num += len(inputs)

        total_loss += loss.item() * len(inputs)
    return total_loss / total_num, total_hit / total_num


# TeacherNet永遠都是Eval mode.
accuracy = []
val_accuracy = []
Train_loss = []
Val_loss = []

teacher_net.eval()
now_best_acc = 0
num_epoch = 400
starttime = 0
for epoch in range(num_epoch):
    student_net.train()
    starttime = time.time()
    train_loss, train_acc = run_epoch(train_dataloader, update=True)
    student_net.eval()
    valid_loss, valid_acc = run_epoch(valid_dataloader, update=False)
    # 存下最好的model。

    torch.save(student_net.state_dict(), 'model.bin')
    print('epoch {:^3d}: {:.3f} sec | train acc: {:6.4f} loss: {:6.4f} | valid acc: {:6.4f} loss {:6.4f}'.format(
        epoch+1, time.time()-starttime, train_acc, train_loss, valid_acc, valid_loss))
    accuracy.append(train_acc)
    Train_loss.append(train_loss)

# 結果存入檔案
accuracy = np.array(accuracy)

Train_loss = np.array(Train_loss)


# result = np.concatenate([accuracy.reshape([1, -1]), val_accuracy.reshape(
#     [1, -1]), Train_loss.reshape([1, -1]), Val_loss.reshape(1, -1)], axis=0)
# result_df = pd.DataFrame(data=result, index=['train_acc',  'val_acc', 'train_loss', 'val_loss'], columns=[
#     'epoch'+str(i)for i in range(1, num_epoch+1)])

# if not os.path.isdir('./result'):
#     os.makedirs('./result')
# result_df.to_csv('./result/acc_loss.csv')

# print("the highest accuracy is at epoch:{}".format(np.argmax(val_accuracy)+1))
# print("the lowest val_loss is at epoch:{}".format(np.argmin(Val_loss)+1))

# plt.plot(accuracy, 'b', label="train accuracy")
# plt.plot(val_accuracy, 'r', label="validation accuracy")

# plt.xlabel("epoch time")
# plt.ylabel("accuracy")
# plt.savefig("./result/acc.png")
# plt.close()

# plt.plot(Train_loss, 'b', label="train loss")
# plt.plot(Val_loss, 'r', label='validation loss')
# plt.xlabel("epoch time")
# plt.ylabel("loss")
# plt.savefig("./result/loss.png")
# plt.close()


''' load test data'''

test_x, test_y = readfile(os.path.join(sys.argv[1], 'testing'))


print('Size of Testing Data:{}'.format(len(test_x)))

# get dataloader

test_set = get_dataset(test_x, test_y, batch_size=32)
test_loader = get_dataloader(test_set, 'testing', batch_size=32)


''' predict'''
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
