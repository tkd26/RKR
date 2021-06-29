# -*- coding: utf-8 -*-
import os
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
from model.utils import load_init_model_state
from data.data_loader import *
from util import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to the config file.')
# parser.add_argument('--resume', type=bool, default=False, help='Resume training.')
# parser.add_argument('--gpu_id', type=int, default='-1', help='gpu id: e.g. 0 1. use -1 for CPU')
opts = parser.parse_args()

conf = get_config(opts.config)
########################################################################
# Setup

writer = tbx.SummaryWriter(log_dir="./results/{}/logs/".format(conf['conf_name']))

model_path = './results/{}/model/'.format(conf['conf_name'])
if not os.path.exists(model_path):
    os.mkdir(model_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

best_score = 0

trainloader_list, testloader_list, classes_list= load_split_cifar100(conf['batch_size'], conf['task_num'])
print('Finished Loading Data')

########################################################################
# train

# define model
net = resnet18(pretrained=True)

# net = resnet_addfc(net, 100)
net = resnet_addfc(net, 10).to(device)
init_net = net

net = resnet18_RKR(pretrained=False, num_classes=10, K=conf['K']).to(device) # best modelをロードするとさらに良いかも
net = load_init_model_state(from_model=init_net, to_model=net) # pretrainモデルで初期化

# loss
criterion = nn.CrossEntropyLoss().to(device)

# optimizer
# optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
# scheduler = MultiStepLR(optimizer, milestones=[50, 100, 125], gamma=0.1)

# train
for task in range(conf['task_num']):
    print('------------------------------')
    print('task{}'.format(task))
    trainloader, testloader, classes = trainloader_list[task], testloader_list[task], classes_list[task]
    if task == 0:
        for name, param in net.named_parameters():
            if 'sfg' in name or 'rg' in name:
                param.requires_grad = False
    else:
        for name, param in net.named_parameters():
            if 'sfg' in name or 'rg' in name:
                param.requires_grad = True
            elif name in ['fc.weight', 'fc.bias']:
                param.requires_grad = True
            elif 'bn' in name or 'downsample.1' in name:
                param.requires_grad = True
            # elif name in ['conv1.weight', 'conv1.bias']:
            #     param.requires_grad = True
            else:
                param.requires_grad = False

    # optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=[50, 100, 125], gamma=0.1)

    for epoch in range(1, conf['epochs'] + 1):  # loop over the dataset multiple times

        # net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
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
        # net.eval()
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