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
SCRIPT_NAME, DATA_DIR, SAVE_PATH, MODEL_NAME, PRETRAINED = argv

print(SCRIPT_NAME)
print(DATA_DIR)



# Define a transform to convert PIL image to a Torch tensor
#Data augmentation added
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

num_classes=2

# Collecting data from the DATA_DIR folder (train)
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
train_data=get_data(DATA_DIR)

if len(train_data)==0:
    print("There is no such dirrectory")
    exit()




BATCH_SIZE=32
train_iter = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)


def train(net, train_iter, trainer, num_epochs):
    loss = nn.CrossEntropyLoss(reduction="mean")
    net.train()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            trainer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().item()
            n += y.shape[0]
            print("Step. time since epoch: {:.3f}. Train acc: {:.3f}. Train Loss: {:.3f}".format(time.time() -  start,
                (y_hat.argmax(axis=1) == y).sum().item() / y.shape[0], l.item()))
        print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, time.time() - start))


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
    if PRETRAINED=="False":
        #model=VGG16(conv_arch, vgg_block, num_classes)
        model=VGG16(num_classes)
        trainer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    else:
        model = tv.models.vgg16(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

#дообучиваем classifier
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096,num_classes))

        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

        trainer = torch.optim.SGD(params_to_update, lr=0.0001, momentum=0.9)



elif MODEL_NAME=='ResNet18':
    if PRETRAINED =="False":
        model = ResNet18(3, ResBlock, outputs=2)
        trainer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    else:
        model = tv.models.resnet18(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        # дообучиваем classifier
        model.fc = nn.Linear(512, num_classes)

        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                    params_to_update.append(param)

        trainer = torch.optim.SGD(params_to_update, lr=0.0001, momentum=0.9)

#тренируем модель

num_epochs=20


train(model, train_iter, trainer, 20)

#сохраняем модель

torch.save(model.state_dict(), SAVE_PATH)


