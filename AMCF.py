from torchvision.datasets import ImageFolder
import sys
import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn
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
import warnings
temp = sys.stdout

# Redirect the output to a file
f = open('outfile/code1.log', 'w')

# After that, use the print function to print the content to a .log file
sys.stdout = f
warnings.filterwarnings("ignore")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_batch_finetune(images, labels, model1, model2, criterion):
    '''
    Run a batch of training and return the training logs of the current batch
    :param images:
    :param labels:
    :return:
    '''
    images = images.to(device)
    labels = labels.to(device)
    outputs1 = model1(images)
    outputs2 = model2(images)
    ###############
    labels_onehot = torch.zeros_like(outputs1)  # Creates a one-hot encoded tensor with the same shape as the outputs
    labels_onehot.scatter_(1, labels.unsqueeze(1), 1)  # The correct category position for the one-hot encoding is assigned a value of 1
    loss = criterion(outputs1, outputs2, labels_onehot.float()) # Calculate a custom MC-loss loss function
    ####################
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    loss.backward()
    optimizer1.step()
    optimizer2.step()
    _,preds = torch.max(outputs1 + outputs2, 1)
    preds = preds.cpu().numpy()
    loss = loss.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    log_train = {}
    log_train['train_loss'] = loss
    log_train['preds'] = preds
    log_train['labels'] = labels

    return log_train
def evaluate_testset(model1, model2, criterion):
    loss_list = []
    labels_list = []
    preds_list = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs1 = model1(images)
            outputs2 = model2(images)

            _,preds = torch.max(outputs1 + outputs2,1)
            preds = preds.cpu().numpy()
            #########################
            labels_onehot = torch.zeros_like(outputs1)  # Creates a one-hot encoded tensor with the same shape as the outputs
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)  # The correct category position for the one-hot encoding is assigned a value of 1
            loss = criterion(outputs1 , outputs2, labels_onehot.float())  # Calculate a custom MC-loss loss function
            ######################
            loss = loss.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            loss_list.append(loss)
            labels_list.extend(labels)
            preds_list.extend(preds)
    log_test = {}

    log_test['test_loss'] = np.mean(loss_list)
    log_test['test_accuracy'] = accuracy_score(labels_list, preds_list)
    log_test['test_precision'] = precision_score(labels_list, preds_list, average = 'macro')
    log_test['test_recall'] = recall_score(labels_list, preds_list, average='macro')
    log_test['test_precision'] = f1_score(labels_list, preds_list, average='macro')

    return log_test

class My_loss(nn.Module):
    def __init__(self):
        super(My_loss, self).__init__()

    def forward(self, outputs1, outputs2, labels):
        # Perform softmax operations on the input
        outputs_softmax1 = torch.softmax(outputs1, dim=1)
        outputs_softmax2 = torch.softmax(outputs2, dim=1)
        # Calculate the cross-entropy loss
        loss1 = -torch.sum(labels * torch.log(outputs_softmax1 + 1e-8)) / labels.size(0)
        loss2 = -torch.sum(labels * torch.log(outputs_softmax2 + 1e-8)) / labels.size(0)
        outputs_mean = torch.mean(torch.stack([outputs1, outputs2], dim=0), dim=0)
        loss = loss1 + loss2 + 0.2 * torch.mean(torch.norm(outputs1-outputs_mean, p=2, dim=1)/2+torch.norm(outputs2-outputs_mean, p=2, dim=1)/2)
        return loss


data_name = ['indoor', 'stf_dog', 'aircraft', 'ucf101', 'omniglot','caltech256-30','caltech256-60']
num_classes = [67, 120, 100, 101, 1623,256,256]

for data_index in [2,0,1,2,3,4,5,6]:
    print(data_name[data_index])

    seed = 0
    setup_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    data_dir = "./datasets/" + data_name[data_index] + '/'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    train_loader = DataLoader(ImageFolder(os.path.join(data_dir, 'train'), transform=train_augs),
                              batch_size, shuffle=True)
    test_loader = DataLoader(ImageFolder(os.path.join(data_dir, 'val'), transform=test_augs),
                             batch_size, shuffle=False, )


    class PSO:

        def __init__(self, D, N, M, p_low, p_up, v_low, v_high, w, c1 = 1.5, c2 = 1.5):
            self.w = w  # Inertia weights
            self.c1 = c1  # Individual learning factors
            self.c2 = c2  # Group learning factor
            self.D = D  # Particle dimension
            self.N = N  # Swarm size, initialization of the number of populations
            self.M = M  # Maximum number of iterations
            self.p_range = [p_low, p_up]  # The extent of the constraint at the particle position
            self.v_range = [v_low, v_high]  # The range of constraints for particle velocity
            self.x = np.zeros((self.N, self.D))  # The position of all particles
            self.v = np.zeros((self.N, self.D))  # The velocity of all particles
            self.p_best = np.zeros((self.N, self.D))  # The optimal position of each particle
            self.g_best = np.zeros((1, self.D))[0]  # The optimal location of the population (global).
            self.p_bestFit = np.zeros(self.N)  # The optimal fit for each particle
            self.g_bestFit = 0.0000001

            # Initialize all individual and global information
            for i in range(self.N):
                for j in range(self.D):
                    self.x[i][j] = random.uniform(self.p_range[0], self.p_range[1])
                    self.v[i][j] = random.uniform(self.v_range[0], self.v_range[1])
                self.p_best[i] = self.x[i]  # Save the historical optimal position of the individual, and the initial default generation 0 is the optimal
                fit = self.fitness(self.p_best[i])
                self.p_bestFit[i] = fit  # The historical optimal adaptation value of the individual is saved
                if fit > self.g_bestFit:  # Find and save the global optimal location and adaptation values
                    self.g_best = self.p_best[i]
                    self.g_bestFit = fit

        def fitness(self,x): #The 'x' here represents the number of layers frozen in multiple source domain models.
            setup_seed(seed)
            pretrained_net1 = models.resnet50(pretrained = True)
            in_features = pretrained_net1.fc.in_features
            pretrained_net1.fc = nn.Linear(in_features, num_classes[data_index])
            pretrained_net1 = pretrained_net1.to(device)
            lr, num_epochs = 0.01, 5

            layer_num = -1
            layer_tune = int(x[0])
            for name, params in pretrained_net1.named_parameters():
                if 'conv' in name:
                    layer_num += 1
                if layer_num>layer_tune or layer_num==layer_tune:
                    break
                else:
                    params.requires_grad = False

            global optimizer1        
            optimizer1 = optim.SGD(filter(lambda p: p.requires_grad, pretrained_net1.parameters()), momentum=0.9, lr=lr, weight_decay=0.0005)
            lr_scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=30, gamma=0.1)

            pretrained_net2 = models.resnet50(pretrained=False)
            pathfile = 'pretrained_models/pre_model_cifar100_resnet50.pth'
            pretrained_net2 = torch.load(pathfile)
            in_features = pretrained_net2.fc.in_features
            pretrained_net2.fc = nn.Linear(in_features, num_classes[data_index])
            pretrained_net2 = pretrained_net2.to(device)

            layer_num = -1
            layer_tune = int(x[1])
            for name, params in pretrained_net2.named_parameters():
                if 'conv' in name:
                    layer_num += 1
                if layer_num>layer_tune or layer_num==layer_tune:
                    break
                else:
                    params.requires_grad = False

            global optimizer2        
            optimizer2 = optim.SGD(filter(lambda p: p.requires_grad, pretrained_net2.parameters()), momentum=0.9, lr=lr, weight_decay=0.0005)
            criterion = My_loss()#nn.CrossEntropyLoss()
            lr_scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=30, gamma=0.1)


            test_accuracy_list = []
            for epoch in range(1, num_epochs + 1):
                start = time.time()
                print(f'Epoch {epoch}/{num_epochs}')
                pretrained_net1.train()
                pretrained_net2.train()
                train_loss_list = []
                train_pre_list = []
                train_label_list = []
                for images, labels in tqdm(train_loader):
                    log_train = train_one_batch_finetune(images, labels, pretrained_net1, pretrained_net2, criterion)
                    train_loss_list.append(log_train['train_loss'])
                    train_pre_list.extend(log_train['preds'])
                    train_label_list.extend(log_train['labels'])
                train_loss = np.mean(train_loss_list)
                train_accuracy = accuracy_score(train_label_list, train_pre_list)
                lr_scheduler1.step()
                lr_scheduler2.step()
                pretrained_net1.eval()
                pretrained_net2.eval()
                log_test = evaluate_testset(pretrained_net1, pretrained_net2, criterion)
                test_accuracy = log_test['test_accuracy']
                test_accuracy_list.append(test_accuracy)
                test_loss = log_test['test_loss']
                print(f'Epoch {epoch}/{num_epochs}, train_accuracy {train_accuracy}, loss {train_loss}, test_accuracy {test_accuracy}, loss {test_loss}, time {time.time() - start}')
            del pretrained_net1
            del pretrained_net2
            del lr_scheduler1
            del lr_scheduler2
            gc.collect()
            max_test_acc = max(test_accuracy_list)
            random.seed() #Unrandom seeds

            np.random.seed() #Unrandom seeds
            fit = max_test_acc
            return fit

        def update(self):
            for i in range(self.N):
                # Update Speed (Core Formula)
                self.v[i] = self.w * self.v[i] + self.c1 * random.uniform(0, 1) * (
                        self.p_best[i] - self.x[i]) + self.c2 * random.uniform(0, 1) * (self.g_best - self.x[i])
                # speed limit
                for j in range(self.D):
                    if self.v[i][j] < self.v_range[0]:
                        self.v[i][j] = self.v_range[0]
                    if self.v[i][j] > self.v_range[1]:
                        self.v[i][j] = self.v_range[1]
                # Update the location
                self.x[i] = self.x[i] + self.v[i]
                # Location Restrictions
                for j in range(self.D):
                    if self.x[i][j] < self.p_range[0]:
                        self.x[i][j] = self.p_range[0]
                    if self.x[i][j] > self.p_range[1]:
                        self.x[i][j] = self.p_range[1]
                # Update individual and global historical optimal locations and adaptation values
                _fit = self.fitness(self.x[i])
                if _fit > self.p_bestFit[i]:
                    self.p_best[i] = self.x[i].copy()
                    self.p_bestFit[i] = _fit.copy()
                if _fit > self.g_bestFit:
                    self.g_best = self.x[i].copy()
                    self.g_bestFit = _fit.copy()

        def pso(self, draw = 1):
            best_fit = []  # Record the optimal fitness for each iteration
            w_range = None
            if isinstance(self.w, tuple):
                w_range = self.w[1] - self.w[0]
                self.w = self.w[1]
            time_start = time.time()  # Record the start time of iterative optimization
            for i in range(self.M):
                self.update()  # Update the main parameters and information
                if w_range:
                    self.w -= w_range / self.M  # The inertia weight decreases linearly
                print("\rIter: {:d}/{:d} fitness: {:.4f} ".format(i, self.M, self.g_bestFit, end = '\n'))
                print('The best individuals are:', self.g_best,'\n')
                best_fit.append(self.g_bestFit.copy())
            time_end = time.time()  # Record the end time of iterative optimization
            print(f'Algorithm takes {time_end - time_start} seconds')  #The total running time of the print algorithm in seconds/s
            return self.g_best



    if __name__ == '__main__':
        low = 0
        up = 48
        pso = PSO(2, 10, 5, low, up, -5, 5, w = (0.7, 1.4))
        #pso(dimension, number of individuals, number of iterations,...
        x=pso.pso()

        #Final fine-tuning phase
        setup_seed(seed)
        pretrained_net1 = models.resnet50(pretrained=True)
        in_features = pretrained_net1.fc.in_features
        pretrained_net1.fc = nn.Linear(in_features, num_classes[data_index])
        pretrained_net1 = pretrained_net1.to(device)
        lr, num_epochs = 0.01, 110

        layer_num = -1
        layer_tune = int(x[0])
        for name, params in pretrained_net1.named_parameters():
            if 'conv' in name:
                layer_num += 1
            if layer_num>layer_tune or layer_num==layer_tune:
                break
            else:
                params.requires_grad = False

        global optimizer1        
        optimizer1 = optim.SGD(filter(lambda p: p.requires_grad, pretrained_net1.parameters()), momentum=0.9, lr=lr, weight_decay=0.0005)
        lr_scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=30, gamma=0.1)

        pretrained_net2 = models.resnet50(weights=None)
        pathfile = 'pretrained_models/pre_model_cifar100.pth'
        pretrained_net2 = torch.load(pathfile)
        in_features = pretrained_net2.fc.in_features
        pretrained_net2.fc = nn.Linear(in_features, num_classes[data_index])
        pretrained_net2 = pretrained_net2.to(device)

        layer_num = -1
        layer_tune = int(x[1])
        for name, params in pretrained_net2.named_parameters():
            if 'conv' in name:
                layer_num += 1
            if layer_num>layer_tune or layer_num==layer_tune:
                break
            else:
                params.requires_grad = False

        global optimizer2        
        optimizer2 = optim.SGD(filter(lambda p: p.requires_grad, pretrained_net2.parameters()), momentum=0.9, lr=lr, weight_decay=0.0005)
        criterion = My_loss()#nn.CrossEntropyLoss()
        lr_scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=30, gamma=0.1)


        test_accuracy_list = []
        for epoch in range(1, num_epochs + 1):
            start = time.time()
            print(f'Epoch {epoch}/{num_epochs}')
            pretrained_net1.train()
            pretrained_net2.train()
            train_loss_list = []
            train_pre_list = []
            train_label_list = []
            for images, labels in tqdm(train_loader):
                log_train = train_one_batch_finetune(images, labels, pretrained_net1, pretrained_net2, criterion)
                train_loss_list.append(log_train['train_loss'])
                train_pre_list.extend(log_train['preds'])
                train_label_list.extend(log_train['labels'])
            train_loss = np.mean(train_loss_list)
            train_accuracy = accuracy_score(train_label_list, train_pre_list)
            lr_scheduler1.step()
            lr_scheduler2.step()
            pretrained_net1.eval()
            pretrained_net2.eval()
            log_test = evaluate_testset(pretrained_net1, pretrained_net2, criterion)
            test_accuracy = log_test['test_accuracy']
            test_accuracy_list.append(test_accuracy)
            test_loss = log_test['test_loss']
            print(f'Epoch {epoch}/{num_epochs}, train_accuracy {train_accuracy}, loss {train_loss}, test_accuracy {test_accuracy}, loss {test_loss}, time {time.time() - start}')

f.close()