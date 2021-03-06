import argparse

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import transforms


def get_mnist_utils(args):
    data_dir = args.data_path

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                   transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_dataset = datasets.MNIST(data_dir, train=False,
                                  transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    criterion = nn.NLLLoss()

    return train_loader, test_loader, criterion

class Network(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return F.log_softmax(x)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

def train(model, train_loader, optimizer, args):
    model.train()
    for i, (input, target) in enumerate(train_loader):
        input = input.reshape(-1, 28 * 28).cuda()
        target = target.cuda()
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0 or i == len(train_loader)-1:
            print(f'Iter[{i}/{len(train_loader)-1}] : Loss {loss.data:.2f}')

def validate(model, val_loader, criterion, epoch, args):
    top1 = AverageMeter()
    losses = AverageMeter()
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            input = input.reshape(-1, 28 * 28).cuda()
            target = target.cuda()
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)

        acc1 = accuracy(output.data, target)[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))

    print(f'Epoch [{epoch}/{args.epochs}]: Acc {top1.avg}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Network Training')
    parser.add_argument('--data_path', default='./data', type=str, help='path to dataset')
    parser.add_argument('--epochs', default=15, type=int, help='number of total epochs to run')
    parser.add_argument('--wd', '--weight_decay', default=1e-3, type=float, help='weight decay')
    parser.add_argument('--m', '--momentum', default=0.99, type=float, help='momentum')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
    parser.add_argument('--print_freq', default=50, type=int, help='print frequency')

    args = parser.parse_args()

    train_loader, val_loader, criterion = get_mnist_utils(args)
    input_size = 784
    output_size = 10
    hidden_size1 = 256
    hidden_size2 = 1024

    model = Network(input_size, hidden_size1, hidden_size2, output_size).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.m, weight_decay=args.wd, nesterov=True)

    for epoch in range(args.epochs):
        train(model, train_loader, optimizer, args)
        validate(model, val_loader, criterion, epoch, args)

    torch.save(model.state_dict(), 'params.pth')

