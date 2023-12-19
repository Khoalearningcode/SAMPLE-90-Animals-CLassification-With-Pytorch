from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import os

data = 'data/animals'
train_dir, val_dir, test_dir = 'data/animals/train', 'data/animals/val', 'data/animals/test'
BATCH_SIZE = 10
NUM_WORKERS = os.cpu_count()
train_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.GaussianBlur(3),
    transforms.ColorJitter(0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def getdata(train_dir, val_dir, test_dir, train_tf, val_tf, batch_size, num_workers):
    train_data = ImageFolder(root=train_dir, transform=train_tf)
    val_data = ImageFolder(root=val_dir, transform=val_tf)
    test_data = ImageFolder(root=test_dir, transform=val_tf)

    train_dataloader = DataLoader(dataset=train_data,
                                  shuffle=True,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  pin_memory=True)
    val_dataloader = DataLoader(dataset=val_data,
                                shuffle=False,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=True)
    test_dataloader = DataLoader(dataset=test_data,
                                 shuffle=False,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 pin_memory=True)
    class_names = train_data.classes
    return train_dataloader, val_dataloader, test_dataloader, class_names


train_dataloader, val_dataloader, test_dataloader, class_names = getdata(train_dir,
                                                                         val_dir,
                                                                         test_dir,
                                                                         train_tf,
                                                                         val_tf,
                                                                         BATCH_SIZE,
                                                                         NUM_WORKERS)

