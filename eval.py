import torch
from torch import nn
import torchvision as tv
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import os
from os import listdir
from sys import argv

#получили параметры файла
SCRIPT_NAME, DATA_DIR, MODEL_NAME, PRETRAINED = argv

# Define a transform to convert PIL image to a Torch tensor
transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

num_classes=2

# Collecting data from the DATA_DIR folder (val)
def get_data(DATA_DIR):
    train_data=[]
    for (root,dirs,files) in os.walk(DATA_DIR, topdown=True):
        class_val=-1
        for name_dir in dirs:
            class_val+=1 #для муравьев -0, для пчел - 1
            for file_name in listdir(DATA_DIR+name_dir+"/"):
                file_name=DATA_DIR+name_dir+"/"+file_name
                image = Image.open(file_name)
                img_tensor = transform(image)
                train_data.append((img_tensor, class_val))
    return train_data


#DATA_DIR="hymenoptera_data/train/"
val_data=get_data(DATA_DIR)



if len(val_data)==0:
    print("There is no such dirrectory")
    exit()




BATCH_SIZE=32
val_iter = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)


def eval(net, val_iter):
    loss = nn.CrossEntropyLoss(reduction='mean')
    net.eval()
    val_l_sum, val_acc_sum, n, start = 0.0, 0.0, 0, time.time()
    for X, y in val_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        val_l_sum += l.item()
        val_acc_sum += (y_hat.argmax(axis=1) == y).sum().item()
        n += y.shape[0]
        print("Batch acc: {:.3f}. batch Loss: {:.3f}".format((y_hat.argmax(axis=1) == y).sum().item() / y.shape[0], l.item()))
    print('Val loss %.4f, Val acc %.3f, time %.1f sec' % (val_l_sum / n, val_acc_sum / n, time.time() - start))


#VGG16 model

class VGG16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3=nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096,num_classes))



    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.classifier(x)
        return x




# Define ResNet block

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)


#ResNet18 model
class ResNet18(nn.Module):
    def __init__(self, in_channels, resblock, outputs=2):
        super().__init__()

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )


        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
                               nn.Flatten(),
                               nn.Linear(512, num_classes))

    def forward(self, x):
        x=self.layer0(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.gap(x)
        x=self.classifier(x)

        return x



#choosing model

if MODEL_NAME=='VGG16':
    model = VGG16(num_classes)
    if PRETRAINED=='False':
        model.load_state_dict(torch.load("models/VGG16/VGG16.pt"),strict=False)  # loads the trained model
    else:
        model.load_state_dict(torch.load("models/VGG16Pretrained/VGG16pretrained.pt"),strict=False)


elif MODEL_NAME=='ResNet18':
    model = ResNet18(3, ResBlock, outputs=2)
    if PRETRAINED =='False':
        model.load_state_dict(torch.load("models/ResNet18/ResNet18.pt"), strict=False)  # loads the trained model
    else:
        model.load_state_dict(torch.load("models/ResNet18Pretrained/ResNet18pretrained.pt"), strict=False)

#тестируем модель


eval(model, val_iter)




