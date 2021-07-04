#coding: utf-8
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data

train_image_dir = "/ImageNet2012/train/"
test_image_dir = "/ImageNet2012/val/"

def load_data(train_batch_size, test_batch_size):
    train_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train = datasets.ImageFolder(train_image_dir, train_transform)
    val = datasets.ImageFolder(train_image_dir, test_transform)
    test = datasets.ImageFolder(test_image_dir, test_transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size, shuffle=True, num_workers=32)
    val_loader = torch.utils.data.DataLoader(val, batch_size=train_batch_size, shuffle=False, num_workers=32)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size, shuffle=False, num_workers=8)

    return train_loader, val_loader, test_loader
