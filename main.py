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
import logging
from tqdm import tqdm

from model.resnet import resnet18, resnet_addfc, resnet34
from model.resnet_RKR import resnet18 as resnet18_RKR
from model.resnet_RKR import resnet34 as resnet34_RKR
from model.resnet_RKR import wide_resnet50_2 as wide_resnet50_RKR
from model.resnet_RKR2 import resnet18 as resnet18_RKR2
from model.resnet_RKR3_1 import resnet18 as resnet18_RKR3_1
from model.resnet_RKR3_2 import resnet18 as resnet18_RKR3_2
from model.LeNet_RKR import LeNet as LeNet_RKR
from model.utils import load_pre_model_state, load_pre_rg_sfg_state, load_pre_fc_state, load_state_dict_from_url
from data.data_loader import *
from util import get_config
from scheduler import WarmupLinearSchedule, WarmupCosineSchedule

import sgd

# DDP
from argparse import ArgumentParser
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

seed = 100
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to the config file.')
# parser.add_argument('--resume', type=bool, default=False, help='Resume training.')
parser.add_argument('--gpu_id', type=str, default=None, help='gpu id: e.g. 0 1. use -1 for CPU')
parser.add_argument('--load_base', type=str, default=None)
parser.add_argument('--start_task', type=int, default=0)
parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank.')
opts = parser.parse_args()

opts.is_master = opts.local_rank == 0
# init
# if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#     rank = int(os.environ["RANK"])
#     world_size = int(os.environ['WORLD_SIZE'])
#     print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
# else:
#     rank = -1
#     world_size = -1
torch.cuda.set_device(opts.local_rank)  
dist.init_process_group(backend='nccl', init_method='env://')


conf = get_config(opts.config)


''' 
-------------------------------------------
設定
-------------------------------------------
'''
#handler2を作成
handler = logging.FileHandler(filename="./logfile/{}.log".format(conf['conf_name']))  #handler2はファイル出力
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))

#loggerにハンドラを設定
logger.addHandler(handler)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

if opts.gpu_id != None:
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id

writer = tbx.SummaryWriter(log_dir="./results/{}/logs/".format(conf['conf_name']))

model_path = './results/{}/model/'.format(conf['conf_name'])
if not os.path.exists(model_path):
    os.mkdir(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(device)
if torch.cuda.is_available():
    gpu_num = torch.cuda.device_count()
    logger.info('gpu_num: {}'.format(gpu_num))
    conf['batch_size'] = conf['batch_size'] * gpu_num

logger.info('K={}'.format(conf['model']['K']))

''' 
-------------------------------------------
データセット
-------------------------------------------
'''
if conf['dataset'] == 'CIFAR100':
    trainloader_list, testloader_list, classes_list= load_split_cifar100(conf['batch_size'], conf['model']['task_num'])
elif conf['dataset'] == 'ImageNet':
    trainloader_list, testloader_list, classes_list= load_split_Imagenet(conf['batch_size'], conf['model']['task_num'])
elif conf['dataset'] == 'VD':
    trainloader_list, testloader_list, classes_list= get_VD_loader(conf['batch_size'], conf['model']['task_num'], opts.local_rank)
logger.info('Finished Loading Data')

''' 
-------------------------------------------
モデルの定義（タスク単体での学習の場合は除く）
事前学習済モデルのロード
-------------------------------------------
'''
conf_model = conf['model']
# conf_basemodel = conf_model.copy()
# conf_basemodel['RG'] = False
# conf_basemodel['SFG'] = False

if 'single' not in conf['conf_name']:
    # resnet18
    if conf_model['name'] == 'resnet-18':
        pre_model_dict = torch.load('/host/space0/takeda-m/jupyter/notebook/RKR/model/resnet18-f37072fd.pth')
        if 'RKR2' in conf['conf_name']:
            net = resnet18_RKR2(pretrained=False, conf_model=conf_model)
        elif 'RKR3_1' in conf['conf_name']:
            net = resnet18_RKR3_1(pretrained=False, conf_model=conf_model)
        elif 'RKR3_2' in conf['conf_name']:
            net = resnet18_RKR3_2(pretrained=False, conf_model=conf_model)
        else:
            net = resnet18_RKR(pretrained=False, conf_model=conf_model)
    
    # resnet34
    elif conf_model['name'] == 'resnet-34':
        pre_model_dict = torch.load('/host/space0/takeda-m/jupyter/notebook/RKR/model/resnet34-b627a593.pth')
        net = resnet34_RKR(pretrained=False, conf_model=conf_model)
    
    # wide-resnet50
    elif conf_model['name'] == 'wide-resnet-50':
        pre_model_dict = torch.load('/host/space0/takeda-m/jupyter/notebook/RKR/model/wide_resnet50_2-95faca4d.pth')
        net = wide_resnet50_RKR(pretrained=False, conf_model=conf_model)

    # pretrainedモデルパラメータのロード
    pre_model_keys = [k for k, v in pre_model_dict.items()]
    model_dict = net.state_dict()
    new_model_dict = {}
    for k, v in model_dict.items():
        if k in pre_model_keys:
            # print(k)
            new_model_dict[k] = pre_model_dict[k]
        else:
            new_model_dict[k] = v
    net.load_state_dict(new_model_dict)

    # パラメータ数の出力
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger.info('parameter:{}'.format(n_parameters))

    n_parameters_weights = sum(p.numel() for name, p in net.named_parameters() if p.requires_grad and 'weights_mat' in name)
    logger.info('weights_mat_list parameter:{}'.format(n_parameters_weights))

# DDP
net = DDP(
        net,
        device_ids=[opts.local_rank]
    )

# net = net.to(device)

''' 
-------------------------------------------
学習設定
-------------------------------------------
'''
# loss
criterion = nn.CrossEntropyLoss().to(device)

if opts.load_base != None:
    state_dict = opts.load_base
    net.load_state_dict(torch.load(state_dict))
    logger.info('model loaded')

''' 
-------------------------------------------
学習
-------------------------------------------
'''
for task in range(opts.start_task, conf['model']['task_num']):
    best_score = 0

    logger.info('------------------------------')
    logger.info('task{}'.format(task))
    trainloader, testloader = trainloader_list[task], testloader_list[task]

    # RG，SFGと最終層以外はフリーズ
    if 'single' in conf['conf_name'] : # タスク単体での学習
        # 今はVDしか用意していない．一つのタスクを学習するごとにモデルをリセットしている．
        if conf['dataset'] == 'VD':
            tasks_class = [1000, 100, 100, 2, 47, 43, 1623, 10, 101, 102]

            if conf_model['name'] == 'resnet-18':
                net = resnet18(num_classes=tasks_class[task]).to(device)
                pre_model_dict = torch.load('/host/space0/takeda-m/jupyter/notebook/RKR/model/resnet34-b627a593.pth')
            elif conf_model['name'] == 'resnet-34':
                net = resnet34(num_classes=tasks_class[task]).to(device)
                pre_model_dict = torch.load('/host/space0/takeda-m/jupyter/notebook/RKR/model/resnet34-b627a593.pth')
            pre_model_keys = [k for k, v in pre_model_dict.items()]
            model_dict = net.state_dict()
            new_model_dict = {}
            for k, v in model_dict.items():
                if k in pre_model_keys:
                    # print(k)
                    new_model_dict[k] = pre_model_dict[k]
                else:
                    new_model_dict[k] = v
            net.load_state_dict(new_model_dict)

            n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
            logger.info('parameter:{}'.format(n_parameters))

    elif 'RKR2' in conf['conf_name']:
        if task == 0:
            for name, param in net.named_parameters():
                if 'F_list' in name:
                    # param.requires_grad = False
                    if name.split('.')[-1] != str(task):
                        param.requires_grad = False
                if 'LM_filter' in name or 'RM_filter' in name:
                    param.requires_grad = False
                if 'fc_list' in name:
                    if name.split('.')[-2] != str(task):
                        param.requires_grad = False
        else:
            for name, param in net.named_parameters():
                param.requires_grad = False
                if 'F_list' in name:
                    if name.split('.')[-1] == str(task):
                        param.requires_grad = True
                if 'LM_filter' in name or 'RM_filter' in name:
                    if name.split('.')[-1] == str(task - 1):
                        param.requires_grad = True
                if 'fc_list' in name:
                    if name.split('.')[-2] == str(task):
                        param.requires_grad = True

    elif 'RKR3' in conf['conf_name']:
        for name, param in net.named_parameters():
            param.requires_grad = False
            if 'F_list' in name or 'LM_list' in name or 'RM_list' in name:
                if name.split('.')[-1] == str(task):
                    param.requires_grad = True
            elif 'fc_list' in name:
                if name.split('.')[-2] == str(task):
                    param.requires_grad = True
            elif 'unc_filt' in name or 'weights_mat' in name:
                if name.split('.')[-1] == str(task):
                    param.requires_grad = True
    
    else:
        for name, param in net.named_parameters():
            param.requires_grad = False
            if 'F_list' in name or 'LM_list' in name or 'RM_list' in name or 'M_list' in name:
                if name.split('.')[-1] == str(task):
                    param.requires_grad = True
            elif 'fc_list' in name:
                if name.split('.')[-2] == str(task):
                    param.requires_grad = True


    for name, param in net.named_parameters():
        if param.requires_grad == True:
            print(name)

    # optimizer
    scheduler = None
    scheduler_VD = None
    t_total = 10000 * (512 // conf['batch_size'])
    if conf_model['name'] == 'resnet-18':
        if conf['dataset'] == 'ImageNet':
            optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
            scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.2)
        elif conf['dataset'] == 'VD':
            # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
            # scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.2)
            
            optimizer = optim.SGD(net.parameters(),
                                lr=3e-2,
                                momentum=0.9,
                                weight_decay=0)
            # VDのschedulerは区別するためにscheduler_VDにする
            scheduler_VD = WarmupCosineSchedule(optimizer, warmup_steps=500, t_total=t_total)

            # optimizer = sgd.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1, momentum=0.9, weight_decay=5.)
            # # optimizer = sgd.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01, momentum=0.9, weight_decay=1.)
            # scheduler = MultiStepLR(optimizer, milestones=[80, 100, 120], gamma=0.1)

        else:
            optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
            scheduler = MultiStepLR(optimizer, milestones=[80, 160, 240], gamma=0.1)

    elif conf_model['name'] == 'resnet-34' or conf_model['name'] == 'wide-resnet-50':
        if conf['dataset'] == 'VD':
            optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
            scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.2)

    elif conf_model['name'] == 'LeNet':
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
        scheduler = MultiStepLR(optimizer, milestones=[100, 200, 300, 400], gamma=0.5)
        # scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.5)

    preparam = 0
    global_step = 0

    for epoch in range(1, conf['epochs'] + 1):  # loop over the dataset multiple times

        net.train()
        dist.barrier()

        running_loss = 0.0
        epoch_iterator = tqdm(trainloader,
                        desc="Training",
                        bar_format="{l_bar}{r_bar}",
                        dynamic_ncols=True,
                        )
                        
        for i, data in enumerate(epoch_iterator):
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

            # VDのschedulerの設定の時だけstepで学習の反復回数を決める
            global_step += 1
            if scheduler is not None:
                scheduler_VD.step()
            if conf['dataset'] == 'VD' and global_step % t_total == 0: 
                break
        
        # 他のノードから集める
        dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
        
        if scheduler is not None:
            scheduler.step()

        # eval
        correct = 0
        total = 0
        net.eval()
        dist.barrier()
        with torch.no_grad():
            epoch_iterator = tqdm(testloader,
                desc="Testing",
                bar_format="{l_bar}{r_bar}",
                dynamic_ncols=True,
                )
            for data in epoch_iterator:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # calculate outputs by running images through the network 
                outputs = net(inputs, task)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # 他のノードから集める
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)

        # save results
        if epoch % 10 == 0 or conf['dataset'] == 'VD':
            epoch_loss = running_loss / (i+1)
            running_loss = 0.0
            writer.add_scalar("train/task{}".format(task), epoch, epoch_loss)

            eval_score = correct / total
            logger.info('task%d [%d] loss train: %.3f eval: %.3f' % (task, epoch, epoch_loss, eval_score))
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

        if conf['dataset'] == 'VD' and global_step % t_total == 0:
            logger.info('best score: {}'.format(best_score))
            break

    logger.info('best score: {}'.format(best_score))

print('Finished Training')

writer.close()

# destrory all processes
dist.destroy_process_group()