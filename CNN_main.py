import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from BatchNorm import BatchNorm
# from mnist_dataloader import get_loader
from cifar10_dataloader import get_loader
from pipeline import Pipeline
from CNN_models import CNN, CNN_BN

# 控制3个实验变量的大小所以一共八组实验：(权重初始化标准差, 学习率, 激活函数(ReLU, Sigmoid) )
# 每组实验分别对有或者没有BN的简单CNN进行训练
# batch_size统一为32，统一训练10个epoch

# 实验0：(1, 0.1, ReLU)
# 实验1：(1, 0.1, Sigmoid)
# 实验2：(1, 2.0, ReLU)
# 实验3：(1, 2.0, Sigmoid)
# 实验4：(9, 0.1, ReLU)
# 实验5：(9, 0.1, Sigmoid)
# 实验6：(9, 2.0, ReLU)
# 实验7：(9, 2.0, Sigmoid)

STD = [0.05,5.]
LR = [0.01, 2.0]
ACTIV = [nn.ReLU, nn.Sigmoid]

def experiment(std,lr,activ_func,BN):
    task_name = f'Loss_{std}_{lr}_{activ_func.__name__}'
    if BN:
        task_name += '_BN'
    print(f'\n#### {task_name} ####\n')
    
    train_loader, test_loader = get_loader(batch_size=64,num_workers=4)
    if BN:
        model = CNN_BN(init_scale=std,activ_func=activ_func)
    else:
        model = CNN(init_scale=std,activ_func=activ_func)
        
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)#, momentum=0.9)
    pipeline = Pipeline(task_name=task_name,
                        log_dir=f'results/{task_name}',
                        model=model,
                        optimizer=optimizer,
                        loss_func=nn.CrossEntropyLoss(),
                        train_loader=train_loader,
                        test_loader=test_loader,
                        epochs=20,
                        cuda=True)
    pipeline.working()

if __name__ == '__main__':
    for std in STD:
        for lr in LR:
            for activ_func in ACTIV:
                for BN in [False, True]:
                    experiment(std,lr,activ_func,BN)