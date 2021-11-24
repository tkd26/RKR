import torch
from torch.functional import split
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import pickle

# DDP
from torch.utils.data.distributed import DistributedSampler

seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

class Mydatasets(torch.utils.data.Dataset):
    def __init__(self, origin, transform, class_id):
        self.transform = transform
        # self.train = train
        # self.dataset_all = torchvision.datasets.CIFAR100(root = path, train = self.train, download = False)
        self.dataset = [[data[0], data[1]] for data in origin if data[1] in class_id]
        for i in range(len(self.dataset)):
            before = self.dataset[i][1]
            self.dataset[i][1] = int((class_id == self.dataset[i][1]).nonzero().squeeze())
        self.datanum = len(self.dataset)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_data, out_label = self.dataset[idx][0], self.dataset[idx][1]
        out_label = out_label % 10 # 全てのラベルを0~9にする

        if self.transform:
            out_data = self.transform(out_data)

        return out_data, out_label

def load_cifar100(batch_size, transform):
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, 
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=2)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transform)                                       
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = (
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'
    )
    return trainloader, testloader, classes


def load_split_cifar100(batch_size, split_num, local_rank=0):
    img_size = 32
    normalize = transforms.Normalize(mean=[0.5074, 0.4867, 0.4411],
                                     std=[0.2011, 0.1987, 0.2025])
                                     
    train_transforms = transforms.Compose([
        transforms.RandomCrop(img_size, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    class_id_list = []
    classes_list = []
    trainset_list = []
    trainloader_list = []
    testset_list = []
    testloader_list = []
    
    classes = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
        'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
        'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
        'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
        'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
        'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
        'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
        'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]

    dataset_train = torchvision.datasets.CIFAR100(root = './data', train = True, download = False)
    dataset_test = torchvision.datasets.CIFAR100(root = './data', train = False, download = False)

    # class_id_list = torch.chunk(torch.Tensor([i for i in range(100)]), split_num)
    class_id_list = torch.chunk(torch.randperm(100), split_num)

    for i in range(split_num):
        # if i > 1:
        #     continue
        classes_list.append([classes[int(id)] for id in class_id_list[i]])

        trainset_list.append(Mydatasets(origin=dataset_train, transform=train_transforms, class_id=class_id_list[i]))
        trainloader_list.append(torch.utils.data.DataLoader(trainset_list[i],
                                                        batch_size=batch_size,
                                                        # shuffle=True,
                                                        sampler=DistributedSampler(trainset_list[i], rank=local_rank),
                                                        num_workers=4))

        testset_list.append(Mydatasets(origin=dataset_test, transform=test_transforms, class_id=class_id_list[i]))
        testloader_list.append(torch.utils.data.DataLoader(testset_list[i],
                                                        batch_size=batch_size,
                                                        # shuffle=False,
                                                        sampler=DistributedSampler(testset_list[i], rank=local_rank),
                                                        num_workers=4))
        print('task{} dataset loaded'.format(i))

    return trainloader_list, testloader_list, classes_list


def load_split_Imagenet(batch_size, split_num, local_rank=0):
    img_size = 72

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                     
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        normalize,
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    class_id_list = []
    classes_list = []
    trainset_list = []
    trainloader_list = []
    testset_list = []
    testloader_list = []
    
    classes = [None] * 1000

    train_path = '/home/yanai-lab/takeda-m/space0/dataset/decathlon-1.0/data/imagenet12/space0/split_train/'
    test_path = '/home/yanai-lab/takeda-m/space0/dataset/decathlon-1.0/data/imagenet12/space0/split_val/'

    # print('loading imagenet...')
    # trainset = datasets.ImageFolder(train_path + '/train')
    # print('loading imagenet...')
    # testset = datasets.ImageFolder(test_path + '/val')
    # print('loaded imagenet')

    # trainset = imagefolder_to_datasets(trainset)
    # testset = imagefolder_to_datasets(testset)

    class_id_list = torch.chunk(torch.Tensor([i for i in range(1000)]), split_num)

    for i in range(split_num):
        classes_list.append([classes[int(id)] for id in class_id_list[i]])

        train_root = train_path + str(i)
        trainset_list.append(datasets.ImageFolder(root=train_root, transform=train_transforms))
        trainloader_list.append(torch.utils.data.DataLoader(trainset_list[i],
                                                            batch_size=batch_size,
                                                            # shuffle=True,
                                                            sampler=DistributedSampler(trainset_list[i], rank=local_rank),
                                                            num_workers=0,
                                                            pin_memory=True))
        
        test_root = test_path + str(i)
        testset_list.append(datasets.ImageFolder(root=test_root, transform=test_transforms))
        testloader_list.append(torch.utils.data.DataLoader(testset_list[i],
                                                            batch_size=batch_size,
                                                            # shuffle=False,
                                                            sampler=DistributedSampler(testset_list[i], rank=local_rank),
                                                            num_workers=0,
                                                            pin_memory=True))
        print('task{} dataset loaded'.format(i))

    return trainloader_list, testloader_list, classes_list


def data_transform(img_size, data_path, name, train=True):
    with open(data_path + 'decathlon_mean_std.pickle', 'rb') as handle:
        dict_mean_std = pickle._Unpickler(handle)
        dict_mean_std.encoding = 'latin1'
        dict_mean_std = dict_mean_std.load()

    means = dict_mean_std[name + 'mean']
    stds = dict_mean_std[name + 'std']

    if name in ['gtsrb', 'omniglot', 'svhn']:  # no horz flip
        transform_train = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    if name in ['gtsrb', 'omniglot', 'svhn']:  # no horz flip
        transform_test = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    else:
        transform_test = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    if train:
        return transform_train
    else:
        return transform_test

def get_VD_loader(batch_size, img_size=32, local_rank=0):
    data_path = '/home/yanai-lab/takeda-m/space0/dataset/decathlon-1.0/data/'
    trainloader_list = []
    testloader_list = []
    classes_list = []
    start_task = 0 # imagenetを飛ばす場合は1にする

    do_task_list = ['imagenet12', 'aircraft', 'cifar100', 'daimlerpedcls', 'dtd', 'gtsrb', 'omniglot', 'svhn', 'ucf101', 'vgg-flowers']
    for i in range(len(do_task_list)):
        task_name = do_task_list[i]
        
        if i < start_task:
            trainloader = []
            testloader = []
        else:
            trainset = torchvision.datasets.ImageFolder(
                data_path + task_name + '/train', transform=data_transform(img_size, data_path, task_name))
            testset = torchvision.datasets.ImageFolder(
                data_path + task_name + '/val', transform=data_transform(img_size, data_path, task_name, train=False))

            trainloader = torch.utils.data.DataLoader(trainset,
                                                    batch_size=batch_size,
                                                    # shuffle=True,
                                                    sampler=DistributedSampler(datasets[x], rank=local_rank),
                                                    num_workers=4, pin_memory=True)
            testloader = torch.utils.data.DataLoader(testset,
                                                    batch_size=batch_size,
                                                    # shuffle=False,
                                                    sampler=DistributedSampler(datasets[x], rank=local_rank),
                                                    num_workers=4, pin_memory=True)

        trainloader_list += [trainloader]
        testloader_list += [testloader]
        print('{} dataset loaded'.format(task_name))

    return trainloader_list, testloader_list, classes_list