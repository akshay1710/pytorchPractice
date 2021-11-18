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

#Created Fully Connected layer
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Hyperparameters
input_size = 784 # 28x28 = 784, size of MNIST images (grayscale)
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1


#Load Data
train_dataset = datasets.MNIST(
    root="dataset/",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)
test_dataset = datasets.MNIST(
    root="dataset/",
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=True
)



#Initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get
        data = data.to(device= device)
        targets = targets.to(device = device)
        #Get to correct shape
        data = data.reshape(data.shape[0], -1)

        #forward
        scores = model(data)
        loss = criterion(scores, targets)

        #backward
        optimizer.zero_grad()
        loss.backward()

        #gradient decent and adam step
        optimizer.step()

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
            x = x.reshape(x.shape[0], -1)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct =(predictions == y).sum()
            num_samples = predictions.size(0)
        print(f"Got {num_correct} /{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")
        
    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)