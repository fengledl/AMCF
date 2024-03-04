
from torchvision.datasets import ImageFolder
import sys
sys.path.append('..')
# from utils import *
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torchvision
import random
import time
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import models
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import gc

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_batch_finetune(images, labels, model):
    '''
    运行一个batch的训练，返回当前batch的训练日志
    :param images:
    :param labels:
    :return:
    '''
    #model = torch.nn.DataParallel(model)
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    _,preds = torch.max(outputs, 1)
    preds = preds.cpu().numpy()
    loss = loss.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    log_train = {}
    log_train['epoch'] = epoch
    # log_train['batch'] = batch_idx
    log_train['train_loss'] = loss
    # log_train['train_accuracy'] = accuracy_score(labels, preds)
    log_train['preds'] = preds
    log_train['labels'] = labels

    return log_train
def evaluate_testset(models):
    loss_list = []
    labels_list = []
    preds_list = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = models(images)

            _,preds = torch.max(outputs,1)
            preds = preds.cpu().numpy()
            loss = criterion(outputs, labels)
            loss = loss.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            loss_list.append(loss)
            labels_list.extend(labels)
            preds_list.extend(preds)
    log_test = {}
    log_test['epoch'] = epoch

    log_test['test_loss'] = np.mean(loss_list)
    log_test['test_accuracy'] = accuracy_score(labels_list, preds_list)
    log_test['test_precision'] = precision_score(labels_list, preds_list, average = 'macro')
    log_test['test_recall'] = recall_score(labels_list, preds_list, average='macro')
    log_test['test_precision'] = f1_score(labels_list, preds_list, average='macro')

    return log_test

setup_seed(0)
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 100

normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
train_augs = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
test_augs = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    normalize
])

train_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR100(root='./cifar_data', train=True,
                                            download=True, transform=train_augs)
                                         ,batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR100(root='./cifar_data', train=False,
                                       download=True, transform=test_augs)
                                        ,batch_size=batch_size, shuffle=False)


pretrained_net = models.resnet50(pretrained=True)
in_features = pretrained_net.fc.in_features
pretrained_net.fc = nn.Linear(in_features, 100)
pretrained_net = pretrained_net.to(device)
# pretrained_params = []
# for name, param in pretrained_net.named_parameters():
#     if 'conv' in name:
#         pretrained_params.append(param.clone())
lr,num_epochs = 0.01,70


optimizer = optim.SGD(pretrained_net.parameters(),momentum=0.9,lr=lr,weight_decay=0.0005)
criterion = nn.CrossEntropyLoss()
lr_scheduler1 = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


for epoch in range(1, num_epochs + 1):
    start = time.time()
    print(f'Epoch {epoch}/{num_epochs}')
    pretrained_net.train()
    train_loss_list = []
    train_pre_list = []
    train_label_list = []
    for images, labels in tqdm(train_loader):
        log_train = train_one_batch_finetune(images, labels, pretrained_net)
        train_loss_list.append(log_train['train_loss'])
        train_pre_list.extend(log_train['preds'])
        train_label_list.extend(log_train['labels'])
    train_loss = np.mean(train_loss_list)
    train_accuracy = accuracy_score(train_label_list, train_pre_list)
    lr_scheduler1.step()
    pretrained_net.eval()
    log_test = evaluate_testset(pretrained_net)
    test_accuracy = log_test['test_accuracy']
    test_loss = log_test['test_loss']
    print(f'Epoch {epoch}/{num_epochs}, train_accuracy {train_accuracy}, loss {train_loss}, test_accuracy {test_accuracy}, loss {test_loss}, time {time.time() - start}')
torch.save(pretrained_net, 'pre_model_cifar100_resnet50.pth')
del pretrained_net
del lr_scheduler1
gc.collect()


