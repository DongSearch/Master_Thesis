from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import os
from torchvision.transforms import ToPILImage
from tqdm import tqdm



def train_pre_processing(data_path, batch_size=128, val_ratio=0.1):
    
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # [0,1]
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) # [-1,1]
    ])

    train_dataset = ImageFolder(data_path,transform=train_transforms)
   
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    return train_loader



def test_pre_processing(data_path, batch_size=128):
    

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])


    test_dataset = ImageFolder(root=data_path, transform=test_transforms)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader
