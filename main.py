from __future__ import print_function
import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore")

dataset = 'cifar10'

torch.manual_seed(3407)
torch.cuda.manual_seed_all(3407)
np.random.seed(3407)
random.seed(3407)

if dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10("./data", train=True, download=True,
                    transform=transforms.Compose([
                    transforms.RandomCrop(32,padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])),
            batch_size=128, shuffle=True, num_workers = 2,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10("./data", train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])),
            batch_size=100, shuffle=False, num_workers = 2,pin_memory=True)

elif dataset == 'cifar100':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100("./data", train=True, download=True,
                         transform=transforms.Compose([
                             transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                         ])),
        batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100("./data", train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])),
        batch_size=100, shuffle=False, num_workers=2, pin_memory=True)

dataloader = {
    "train": train_loader,
    "test": test_loader
}

def train(model,model_name):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler_lr = torch.optim.lr_scheduler.StepLR(optimizer,30,gamma=0.1)
    model.cuda()
    best_acc = 0
    for epoch in range(210):
        print('Epoch {}'.format(epoch + 1))
        print('当前学习率 {}'.format(scheduler_lr.get_lr()[0]))
        print('-' * 10)
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for batch_idx,(inputs,labels) in enumerate(dataloader[phase]):
                inputs = inputs.cuda()
                labels = labels.cuda()
                inputs, labels = Variable(inputs), Variable(labels)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = F.cross_entropy(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                if batch_idx % 128 == 0 and phase == 'train' :
                    print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(inputs), len(dataloader[phase].dataset),
                               100. * batch_idx / len(dataloader[phase]), loss.item()))
                elif batch_idx % 100 == 0 and phase == 'test':
                    print('Test Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(inputs), len(dataloader[phase].dataset),
                               100. * batch_idx / len(dataloader[phase]), loss.item()))
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataloader[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloader[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model, "model save path")
        scheduler_lr.step()
    return model

def test(model):
    model.eval()
    test_dataloader = dataloader['test']
    running_corrects = 0
    for batch_idx,(inputs,labels) in enumerate(test_dataloader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
    epoch_acc = running_corrects.double() / len(test_dataloader.dataset)
    print('模型准确率: {:.4f}'.format(epoch_acc))