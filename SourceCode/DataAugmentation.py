import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from CustomeDataSet import *

# Load Data
my_transforms = transforms.Compose(
    [  # Compose makes it possible to have many transforms
        transforms.ToPILImage(),
        #transforms.Resize((36, 36)),  # Resizes (32,32) to (36,36)
        #transforms.RandomCrop((32, 32)),  # Takes a random (32,32) crop
        transforms.ColorJitter(brightness=0.5),  # Change brightness of image
        transforms.RandomRotation(degrees=45),  # Perhaps a random rotation from -45 to 45 degrees
        transforms.RandomHorizontalFlip(p=0.5),  # Flips the image horizontally with probability 0.5
        transforms.RandomVerticalFlip(p=0.05),  # Flips image vertically with probability 0.05
        transforms.RandomGrayscale(p=0.2),  # Converts to grayscale with probability 0.2
        transforms.ToTensor(), # Finally converts PIL image to tensor so we can train w. pytorch

        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Note: these values aren't optimal
    ]
)

dataset = CatsAndDogsDataset(csv_file='cats_dogs.csv', root_dir='cats_dogs_resized', transform=my_transforms)

img_num = 0
for _ in range(30):
    for img, label in dataset:
        save_image(img , 'DataStore/img' + str(img_num) + '.png')
        img_num += 1
