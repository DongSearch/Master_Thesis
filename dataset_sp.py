from torchvision import transforms
from torchvision.datasets import ImageFolder, CIFAR10
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import os
from torchvision.transforms import ToPILImage
from tqdm import tqdm



def train_val_pre_processing(data_path, batch_size=128, val_ratio=0.1):
    
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # [0,1]
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) # [-1,1]
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])


    full_dataset = ImageFolder(root=data_path)

    indicies = list(range(len(full_dataset)))
    #for stratify
    labels = [y for _, y in full_dataset.samples]

    train_idx, val_idx = train_test_split(
        indicies,
        test_size=val_ratio,
        random_state= 42,
        stratify=labels,
        shuffle=True
    )

    train_dataset = Subset(
        ImageFolder(data_path, transform=train_transforms), train_idx
    )
    
    val_dataset = Subset(
        ImageFolder(data_path, transform=val_transforms), val_idx
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader



def test_pre_processing(data_path, batch_size=128):
    

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])


    test_dataset = ImageFolder(root=data_path, transform=test_transforms)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader
