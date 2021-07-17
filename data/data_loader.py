import torch
from torch.functional import split
import torchvision
import torchvision.transforms as transforms

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Mydatasets(torch.utils.data.Dataset):
    def __init__(self, origin, transform, class_id):
        self.transform = transform
        # self.train = train
        # self.dataset_all = torchvision.datasets.CIFAR100(root = path, train = self.train, download = False)
        self.dataset = [data for data in origin if data[1] in class_id]
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
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


def load_split_cifar100(batch_size, split_num):
    normalize = transforms.Normalize(mean=[0.5074, 0.4867, 0.4411],
                                     std=[0.2011, 0.1987, 0.2025])
                                     
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
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
    ]

    dataset_train = torchvision.datasets.CIFAR100(root = './data', train = True, download = False)
    dataset_test = torchvision.datasets.CIFAR100(root = './data', train = False, download = False)

    class_id_list = torch.chunk(torch.Tensor([i for i in range(100)]), split_num)

    for i in range(split_num):
        if i > 1:
            continue
        classes_list.append([classes[int(id)] for id in class_id_list[i]])

        trainset_list.append(Mydatasets(origin=dataset_train, transform=train_transforms, class_id=class_id_list[i]))
        trainloader_list.append(torch.utils.data.DataLoader(trainset_list[i], batch_size=batch_size, shuffle=True, num_workers=2))

        testset_list.append(Mydatasets(origin=dataset_test, transform=test_transforms, class_id=class_id_list[i]))
        testloader_list.append(torch.utils.data.DataLoader(testset_list[i], batch_size=batch_size, shuffle=False, num_workers=2))
        print('task{} dataset loaded'.format(i))

    return trainloader_list, testloader_list, classes_list