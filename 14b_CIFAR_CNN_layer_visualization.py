# -*- coding: utf-8 -*-
"""
YouTube channel: Python Engineer
tutorials for using pytorch
https://youtu.be/pDdP0TFzsoQ

Helper script to visualize the action of the pooling
layers in the Convolutional Neural Net (CNN)
using the CIFAR-10 dataset of 60,000 32x32 px color images 
of the ten classes [airplane, automobile, ..., truck]

https://www.cs.toronto.edu/~kriz/cifar.html
Created on Sun Oct  4 11:51:17 2020

@author: Doug
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#### device config
device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

#### hyperparameters
num_epochs = 0
batch_size = 4
learning_rate = 0.001
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')


### load and transform the dataset
# dataset is PILI images w/range [0,1]
# xform them to Tensors of range [-1,1]
transform = transforms.Compose([transforms.ToTensor(), 
                        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                    download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                    download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, 
                                    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, 
                                    batch_size=batch_size, shuffle=True)


####  view some of the images
def imshow(img):
    img = img / 2.0 + 0.5  # unnormalize [-1,1]->[0,1]
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
    
# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))


#### visualize the action of the pooling and convolution layers
conv1 = nn.Conv2d(3, 6, 5) 
pool = nn.MaxPool2d(2, 2)
conv2 = nn.Conv2d(6,16,5)
print(f'original batch Tensor shape = {images.shape}')
x = conv1(images)
print(f'batch tensor shape after 1st convolution layer = {x.shape}')
x = pool(x)
print(f'batch tensor shape after 1st pooling layer = {x.shape}')
x = conv2(x)
print(f'batch tensor shape after 2nd convolution layer = {x.shape}')
x = pool(x)
print(f'batch tensor shape after 2nd pooling layer = {x.shape}')



