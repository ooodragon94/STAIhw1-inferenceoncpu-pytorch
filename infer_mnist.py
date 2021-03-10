import argparse
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import transforms

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

def validate(model, val_loader, criterion, epoch, args):
    top1 = AverageMeter()
    losses = AverageMeter()
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            input = input.reshape(-1, 28 * 28).cuda()
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)

        acc1 = accuracy(output.data, target)[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))

    print(f'Epoch [{epoch}/{args.epochs}]: Acc {top1.avg}')

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

def accuracy(output, target):
    pred = np.argmax(output, axis=1)
    num_correct = sum(np.equal(pred, target))
    res = num_correct / len(target)
    return res

def np_relu(m):
    m[m < 0] = 0
    return m
def np_log_softmax(x):
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / e_x.sum())

class NumPyNetwork:
    def __init__(self, fc1, b1, fc2, b2, fc3, b3):
        self.fc1 = fc1.cpu().numpy()
        self.fc2 = fc2.cpu().numpy()
        self.fc3 = fc3.cpu().numpy()
        self.b1 = b1.cpu().numpy()
        self.b2 = b2.cpu().numpy()
        self.b3 = b3.cpu().numpy()
    def forward(self, x):
        x = np.einsum('kj,ij->ik', self.fc1, x)
        x = np.sum([self.b1, x])
        x = np_relu(x)
        x = np.einsum('kj,ij->ik', self.fc2, x)
        x = np.sum([self.b2, x])
        x = np_relu(x)
        x = np.einsum('kj,ij->ik', self.fc3, x)
        x = np.sum([self.b3, x])
        x = np_log_softmax(x)
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Network Training')
    parser.add_argument('--data_path', default='./data', type=str, help='path to dataset')
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers')
    parser.add_argument('--batch_sizes', default=[64,128], nargs='*')
    parser.add_argument('--print_freq', default=50, type=int, help='print frequency')

    args = parser.parse_args()

    params = torch.load('params.pth')

    input_size = 784
    output_size = 10
    hidden_size1 = 256
    hidden_size2 = 1024

    model = Network(input_size, hidden_size1, hidden_size2, output_size)
    model.load_state_dict(params)
    np_model = NumPyNetwork(params['fc1.weight'], params['fc1.bias'], params['fc2.weight'],
                            params['fc2.bias'], params['fc3.weight'], params['fc3.bias'])

    inp = np.ones((1,784), dtype=float)
    out1 = np_model.forward(inp)
    out2 = model(torch.from_numpy(inp).float())

    # changes str to int
    batch_sizes = args.batch_sizes
    batch_sizes = [int(batch_size) for batch_size in args.batch_sizes]

    top1 = AverageMeter()

    for batch_size in batch_sizes:
        print(f'testing with batch size: {batch_size}')
        args.batch_size = batch_size
        train_loader, val_loader, criterion = get_mnist_utils(args)
        np_model = NumPyNetwork(params['fc1.weight'], params['fc1.bias'], params['fc2.weight'],
                                params['fc2.bias'], params['fc3.weight'], params['fc3.bias'])
        tic = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.reshape(-1, 28 * 28)
            output = np_model.forward(input)
            acc1 = accuracy(output, target)
            top1.update(acc1, input.size(0))
            if i % args.print_freq == 0:
                print(f'Iter[{i}/{len(val_loader)}]: Acc {top1.avg * 100 :.4f}%')
        toc = time.time()

        print(f'total seconds taken: \033[93m{toc-tic:.6f}\033[0m')
        print(f'final accuracy: \033[93m{top1.avg * 100 :.4f}%\033[0m')