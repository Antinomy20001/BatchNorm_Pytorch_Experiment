import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from BatchNorm import BatchNorm
from cifar10_dataloader import get_loader
from pipeline import Pipeline

from model import AlexNet as Model
from model_bn import AlexNet_BN as Model_BN
###### with BN

pipeline.working()


##### without BN
train_loader, test_loader = get_loader(batch_size=128,num_workers=1)
model = Model()
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.01, momentum=0.9,  weight_decay=5e-2)
pipeline = Pipeline(task_name='alexnet',
                    log_dir='alexnet',
                    model=model,
                    optimizer=optimizer,
                    loss_func=nn.CrossEntropyLoss(),
                    train_loader=train_loader,
                    test_loader=test_loader,
                    epochs=5,
                    cuda=True)
pipeline.working()



