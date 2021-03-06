import argparse
import time
import numpy as np
import torch
from torch import nn
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
        x = np.einsum('j,ij->ij', self.b1, x)
        x = np.einsum('kj,ij->ik', self.fc2, x)
        x = np.einsum('j,ij->ij', self.b2, x)
        x = np.einsum('kj,ij->ik', self.fc3, x)
        x = np.einsum('j,ij->ij', self.b3, x)
        x = np_log_softmax(x)
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Network Training')
    parser.add_argument('--data_path', default='./data', type=str, help='path to dataset')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
    parser.add_argument('--print_freq', default=50, type=int, help='print frequency')

    args = parser.parse_args()

    params = torch.load('params.pth')
    batch_sizes = [1, 32]
    for batch_size in batch_sizes:

        args.batch_size = batch_size
        train_loader, val_loader, criterion = get_mnist_utils(args)
        np_model = NumPyNetwork(params['fc1.weight'], params['fc1.bias'], params['fc2.weight'],
                                params['fc2.bias'], params['fc3.weight'], params['fc3.bias'])
        tic = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.reshape(-1, 28 * 28)
            np_model.forward(input)
        toc = time.time()

        print(f'total seconds taken: {toc-tic}')