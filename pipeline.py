import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter

class Pipeline:
    def __init__(self, task_name='Task',
                 log_dir='task/',
                 seed=int('0510'),
                 cuda=False,
                 model=None,
                 optimizer=None,
                 loss_func=None,
                 epochs=10,
                 log_interval=50,
                 train_loader=None,
                 test_loader=None):

        self.task_name = task_name
        self.log_dir = log_dir
        self.writer = SummaryWriter(self.log_dir)
        self.seed = seed
        self.cuda = cuda

        self.model = model
        if self.cuda:
            self.model = self.model.cuda()
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.epochs = epochs
        self.log_interval = log_interval

        self.train_loader = train_loader
        self.test_loader = test_loader

        torch.manual_seed(self.seed)

    def train(self, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_func(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx *
                    len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.data.item()))
                niter = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar(self.task_name+'-Train/Loss', loss.data.item(), niter)

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in self.test_loader:
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = self.model(data)
            # sum up batch loss
            test_loss += self.loss_func(output,target).data.item()
            # get the index of the max log-probability
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()

        test_loss /= len(self.test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

    def working(self):
        for epoch in range(1, self.epochs + 1):
            self.train(epoch)
            self.test()