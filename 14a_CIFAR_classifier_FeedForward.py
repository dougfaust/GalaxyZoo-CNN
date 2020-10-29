# -*- coding: utf-8 -*-
"""
YouTube channel: Python Engineer
tutorials for using pytorch
https://youtu.be/pDdP0TFzsoQ

Created on Sat Oct  3 11:40:58 2020

@author: Doug

My attempt to write a Feed-Forward Neural Net
using the CIFAR-10 dataset of 60,000 32x32 px color images 
of the ten classes [airplane, automobile, ..., truck]

https://www.cs.toronto.edu/~kriz/cifar.html

"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


#### set or configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#### hyper parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001
show_sample = True

input_size = 3072 # 32x32 px w/ RGB channels
hidden_size = 500
num_classes = 10

classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

#### import dataset
# tranform to normalize images from PILI range [0,1]->[-1,1]
transform = transforms.Compose([transforms.ToTensor(), 
                        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_data = torchvision.datasets.CIFAR10(root='.\data', train=True, 
                        transform=transform, download=True)
test_data = torchvision.datasets.CIFAR10(root='.\data', train=False, 
                        transform=transform, download=False)

#### ingest dataset as batches
# note that you have problems with num_workers / multithreading on this machine
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                        shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                        shuffle=True)

examples = iter(train_loader)

images, labels = examples.next()
# print(samples.shape, labels.shape)

####  view some of the images
def imshow(img):
    img = img / 2.0 + 0.5  # unnormalize [-1,1]->[0,1]
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

# show images
if show_sample: imshow(torchvision.utils.make_grid(images[:5,:,:,:]))

#### define neural net
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        super(NeuralNet, self).__init__()
        
        self.l1 = nn.Linear(input_size, hidden_size)
        self.activ_fn = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
        # no final activation fn in multiclass CrossEntropyLoss has softmax
        
    def forward(self, x):
        out = self.l1(x)
        out = self.activ_fn(out)
        out = self.l2(out)
        return out
   

model = NeuralNet(input_size, hidden_size, num_classes)

    
#### define loss and optimizer functions
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                    
#### training loop
total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        # reshape data and send to device
        images = images.reshape(-1, input_size).to(device)
        labels.to(device)

        # forward
        outputs = model.forward(images)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # print progress
        if (i+1)%100 == 0:
            print(f'epoch = {epoch+1} / {num_epochs}, step = {i+1} / {total_steps}, loss = {loss.item():.4f}')
   
print('\n =====  Finished Training  ===== \n')
        
#### accuracy and results
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_correct_class = [0 for i in range(num_classes)]
    n_samples_class = [0 for i in range(num_classes)]
    
    for images, labels in test_loader:
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        # value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predictions[i]
            if (label == pred):
                n_correct_class[label] += 1
            n_samples_class[label] += 1
        
    acc = 100.0 * n_correct / n_samples
    print(f'accuracy of model = {acc}')
    
    for i in range(num_classes):
        acc = 100.0 * n_correct_class[i] / n_samples_class[i]
        print(f'Accuracy of {classes[i]} class: {acc} %')