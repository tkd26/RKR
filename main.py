# -*- coding: utf-8 -*-
import os
import sys
import argparse
from data.data_loader import load_cifar100
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import tensorboardX as tbx

from model.resnet import resnet18, resnet_addfc
from model.resnet_RKR import resnet18 as resnet18_RKR
from model.LeNet_RKR import LeNet as LeNet_RKR
from model.utils import load_pre_model_state, load_pre_rg_sfg_state, load_pre_fc_state
from data.data_loader import *
from util import get_config

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to the config file.')
# parser.add_argument('--resume', type=bool, default=False, help='Resume training.')
parser.add_argument('--gpu_id', type=str, default='0', help='gpu id: e.g. 0 1. use -1 for CPU')
parser.add_argument('--load_base', type=str, default=None)
opts = parser.parse_args()

conf = get_config(opts.config)
########################################################################
# Setup

os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id

writer = tbx.SummaryWriter(log_dir="./results/{}/logs/".format(conf['conf_name']))

model_path = './results/{}/model/'.format(conf['conf_name'])
if not os.path.exists(model_path):
    os.mkdir(model_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

best_score = 0

if conf['dataset'] == 'CIFAR100':
    trainloader_list, testloader_list, classes_list= load_split_cifar100(conf['batch_size'], conf['model']['task_num'])
elif conf['dataset'] == 'ImageNet':
    trainloader_list, testloader_list, classes_list= load_split_Imagenet(conf['batch_size'], conf['model']['task_num'])
print('Finished Loading Data')

########################################################################
# train

# define model

conf_model = conf['model']
# conf_basemodel = conf_model.copy()
# conf_basemodel['RG'] = False
# conf_basemodel['SFG'] = False

if conf_model['name'] == 'resnet-18':
    init_net = resnet18(pretrained=True)

    init_net = resnet_addfc(init_net, 10).to(device)
    # net = resnet18_RKR(pretrained=False, conf_model=conf_basemodel).to(device) # best modelをロードするとさらに良いかも
    net = resnet18_RKR(pretrained=False, conf_model=conf_model).to(device) # best modelをロードするとさらに良いかも
    net = load_pre_model_state(from_model=init_net, to_model=net) # pretrainモデルで初期化
elif conf_model['name'] == 'LeNet':
    net = LeNet_RKR(conf_model=conf_model).to(device)

# loss
criterion = nn.CrossEntropyLoss().to(device)

start_task = 0
if opts.load_base != None:
    state_dict = opts.load_base
    net.load_state_dict(torch.load(state_dict))
    start_task = 1
    print('model loaded')

# train
for task in range(start_task, conf['model']['task_num']):
    print('------------------------------')
    print('task{}'.format(task))
    trainloader, testloader = trainloader_list[task], testloader_list[task]
    if task == 0:
        for name, param in net.named_parameters():
            if 'F_list' in name or 'LM_list' in name or 'RM_list' in name or 'M_list' in name:
                # param.requires_grad = False
                if name.split('.')[-1] != str(task):
                    param.requires_grad = False
            if 'fc_list' in name:
                if name.split('.')[-2] != str(task):
                    param.requires_grad = False
    else:
        # if task == 1:
        #     base_net = net
        #     net = resnet18_RKR(pretrained=False, conf_model=conf_model).to(device)
        #     net = load_pre_model_state(from_model=base_net, to_model=net)

        net = load_pre_rg_sfg_state(net, task)
        net = load_pre_fc_state(net, task)

        for name, param in net.named_parameters():
            param.requires_grad = False
            if 'F_list' in name or 'LM_list' in name or 'RM_list' in name or 'M_list' in name:
                if name.split('.')[-1] == str(task):
                    param.requires_grad = True
            elif 'fc_list' in name:
                if name.split('.')[-2] == str(task):
                    param.requires_grad = True
            # elif 'LM' in name or 'RM' in name:
            #     param.requires_grad = True

    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)

    # optimizer
    if conf_model['name'] == 'resnet-18':
        if conf['dataset'] == 'ImageNet':
            optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
            scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.2)
        else:
            optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
            scheduler = MultiStepLR(optimizer, milestones=[50, 100, 125], gamma=0.1)
    elif conf_model['name'] == 'LeNet':
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
        scheduler = MultiStepLR(optimizer, milestones=[100, 200, 300, 400], gamma=0.5)
        # scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.5)

    preparam = 0

    for epoch in range(1, conf['epochs'] + 1):  # loop over the dataset multiple times

        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # if task != -1:
            #     param = net.conv1.LM_list[0].clone()
            #     if torch.any(preparam != param):
            #         preparam = param
            #         # print(param)

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs, task)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        scheduler.step()

        # eval
        correct = 0
        total = 0
        net.eval()
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # calculate outputs by running images through the network 
                outputs = net(inputs, task)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # save results
        if epoch % 10 == 0:
            epoch_loss = running_loss / (i+1)
            running_loss = 0.0
            writer.add_scalar("train/task{}".format(task), epoch, epoch_loss)

            eval_score = correct / total
            print('task%d [%d] loss train: %.3f eval: %.3f' % (task, epoch, epoch_loss, eval_score))
            writer.add_scalar("eval/task{}".format(task), epoch, eval_score)

            # save trained model
            task_model_path = './results/{}/model/{}'.format(conf['conf_name'], task)
            if not os.path.exists(task_model_path):
                os.mkdir(task_model_path)
            
            model_path_name = '{}/{:06}.pth'.format(task_model_path, epoch)
            torch.save(net.state_dict(), model_path_name)

            if best_score < eval_score:
                bestmodel_path_name = '{}/best_model.pth'.format(task_model_path, task)
                torch.save(net.state_dict(), bestmodel_path_name)
                best_score = eval_score

print('Finished Training')

writer.close()

########################################################################
# eval samples

# dataiter = iter(testloader)
# images, labels = dataiter.next()
# images, labels = images.to(device), labels.to(device)

# # print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# # define model
# net = resnet18().to(device)
# net = resnet_addfc(net, 100)
# # load trained model
# net.load_state_dict(torch.load(PATH))

# outputs = net(images)

# _, predicted = torch.max(outputs, 1)

# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                               for j in range(4)))

########################################################################
# eval score

# correct = 0
# total = 0

# with torch.no_grad():
#     for data in testloader:
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)
#         # calculate outputs by running images through the network 
#         outputs = net(inputs)
#         # the class with the highest energy is what we choose as prediction
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print('Accuracy of the network on the 10000 test images: %d %%' % (
#     100 * correct / total))

########################################################################
# eval class score

# # prepare to count predictions for each class
# correct_pred = {classname: 0 for classname in classes}
# total_pred = {classname: 0 for classname in classes}

# # again no gradients needed
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         images, labels = images.to(device), labels.to(device)
#         outputs = net(images)    
#         _, predictions = torch.max(outputs, 1)
#         # collect the correct predictions for each class
#         for label, prediction in zip(labels, predictions):
#             if label == prediction:
#                 correct_pred[classes[label]] += 1
#             total_pred[classes[label]] += 1

  
# # print accuracy for each class
# for classname, correct_count in correct_pred.items():
#     accuracy = 100 * float(correct_count) / total_pred[classname]
#     print("Accuracy for class {:5s} is: {:.1f} %".format(classname, 
#                                                    accuracy))