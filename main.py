import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

import argparse
import torch
use_cuda = torch.cuda.is_available()

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from mnist_code.tensorboard import SummaryWriterManager

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def test(model, device, dataloader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
    test_loss /= len(dataloader.dataset)
    model.train()
    return test_loss

def main():
    # Training settings
    #testset = datasets.MNIST(DATADIR, train=False, download=False, transform=transform)
    #test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
    lr =1.0
    boarddir = './board'
    batch_size = 10000

    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    DATADIR = "./mnist_data/"
    trainset = datasets.MNIST(DATADIR, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    ################################################
    boardsubdir = boarddir+'/lr{}/'.format(lr)

    tensorboardwrite = SummaryWriterManager(boardsubdir)
    ################################################

    for epoch in range(1, 15):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
        train_loss = test(model, device, train_loader)
        print("Train")
        print('Loss: {:.4f}, Accuracy : {:.4f}'.format(train_loss, train_acc))
        write_content = {'train_loss' : train_loss}
        tensorboardwrite.add_summaries(write_content, global_step=epoch)

if __name__ == '__main__':
    main()
