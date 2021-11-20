import torch
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F # All functions that don't have any parameters
from torch.utils.data import DataLoader # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets # Has standard datasets we can import in a nice and easy way
import torchvision.transforms as transforms # Transformations we can perform on our dataset

import torchvision
from torch.utils.data import DataLoader # data management
import torchvision.datasets as datasets # standard datasets
import torchvision.transforms as transforms # data processing
import sys

#Created Fully Connected layer
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load pretrain  model & modify it
model =  torchvision.models.vgg16(pretrained=True)
print(model)
for param in model.parameters():
    param.requires_grad =  False
model.avgpool =Identity()
model.classifier = nn.Sequential(nn.Linear(512,100),
                                 nn.ReLU(),
                                 nn.Linear(100,10))
model.to(device)
print(model)




#Hyperparameters
in_channel =  1
num_classes = 10
learning_rate = 1e-3
batch_size = 64
num_epochs = 10
load_model = True


#Load Data
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True,)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True,)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)



#Initialize network


#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



#Train Network
for epoch in range(num_epochs):
    lossed = []


    for batch_idx, (data, targets) in enumerate(train_loader):

        # Get
        data = data.to(device= device)
        targets = targets.to(device = device)
        #Get to correct shape


        #forward
        scores = model(data)
        loss = criterion(scores, targets)
        lossed.append(loss.item())



        #backward
        optimizer.zero_grad()
        loss.backward()

        #gradient decent and adam step
        optimizer.step()
    print(loss)


#Check accuracy to training & test to see how good our model
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("checking accuracy in tranining data")
    else:
        print("checking accuracy in testing data")
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct =(predictions == y).sum()
            num_samples = predictions.size(0)
        print(f"Got {num_correct} /{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")
        print("working on device")
        print("ready")
    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)