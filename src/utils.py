# -*- coding: utf-8 -*-
import os
import logging
import numpy as np
import time
import random
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torch.nn as nn

from PIL import Image, ImageFilter


class GaussianBlur(object):
    """Gaussian blur augmentation: https://github.com/facebookresearch/moco/"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def load_finetune(base_encoder, args):
    # 加载微调好的模型，直接测试

    print("\n\nLoading the model: {}\n\n".format(args.load_checkpoint_dir_finetune))

    # Load the pretrained model
    checkpoint = torch.load(args.load_checkpoint_dir_finetune)
    # Load the encoder parameters
    base_encoder.load_state_dict(checkpoint['base_encoder'])

    return base_encoder

def load_finetune_knn(base_encoder, args):
    # 加载微调好的模型，直接测试

    print("\n\nLoading the model: {}\n\n".format(args.load_checkpoint_dir_finetune))

    # Load the pretrained model
    checkpoint = torch.load(args.load_checkpoint_dir_finetune)
    # Load the encoder parameters
    # base_encoder.load_state_dict(checkpoint['base_encoder'])

    state_dict = checkpoint['base_encoder']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
            # remove prefix
            state_dict[k[len("encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    # Load the encoder parameters
    base_encoder.load_state_dict(state_dict, strict=False)

    return base_encoder


def load_moco(base_encoder, args):
    """ Loads the pre-trained MoCo model parameters.

        Applies the loaded pre-trained params to the base encoder used in Linear Evaluation,
         freezing all layers except the Linear Evaluation layer/s.

    Args:
        base_encoder (model): Randomly Initialised base_encoder.

        args (dict): Program arguments/commandline arguments.
    Returns:
        base_encoder (model): Initialised base_encoder with parameters from the MoCo query_encoder.
    """
    print("\n\nLoading the model: {}\n\n".format(args.load_checkpoint_dir))

    # Load the pretrained model
    checkpoint = torch.load(args.load_checkpoint_dir, map_location="cpu")

    # rename moco pre-trained keys
    state_dict = checkpoint['moco']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
            # remove prefix
            state_dict[k[len("encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    # Load the encoder parameters
    base_encoder.load_state_dict(state_dict, strict=False)

    return base_encoder


def load_sup(base_encoder, args):
    """ Loads the pre-trained supervised model parameters.

        Applies the loaded pre-trained params to the base encoder used in Linear Evaluation,
         freezing all layers except the Linear Evaluation layer/s.

    Args:
        base_encoder (model): Randomly Initialised base_encoder.

        args (dict): Program arguments/commandline arguments.
    Returns:
        base_encoder (model): Initialised base_encoder with parameters from the supervised base_encoder.
    """
    print("\n\nLoading the model: {}\n\n".format(args.load_checkpoint_dir))

    # Load the pretrained model
    checkpoint = torch.load(args.load_checkpoint_dir)

    # Load the encoder parameters
    base_encoder.load_state_dict(checkpoint['encoder'])

    # freeze all layers but the last fc
    for name, param in base_encoder.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False

    # init the fc layer
    init_weights(base_encoder)

    return base_encoder


def init_weights(m):
    '''Initialize weights with zeros
    '''

    # init the fc layer
    m.fc.weight.data.normal_(mean=0.0, std=0.01)
    m.fc.bias.data.zero_()


class CustomDataset(Dataset):
    """ Creates a custom pytorch dataset.

        - Creates two views of the same input used for unsupervised visual
        representational learning. (SimCLR, Moco, MocoV2)

    Args:
        data (array): Array / List of datasamples

        labels (array): Array / List of labels corresponding to the datasamples

        transforms (Dictionary, optional): The torchvision transformations
            to make to the datasamples. (Default: None)

        target_transform (Dictionary, optional): The torchvision transformations
            to make to the labels. (Default: None)

        two_crop (bool, optional): Whether to perform and return two views
            of the data input. (Default: False)

    Returns:
        img (Tensor): Datasamples to feed to the model.

        labels (Tensor): Corresponding lables to the datasamples.
    """

    def __init__(self, data, labels, transform=None, target_transform=None, two_crop=False):

        # shuffle the dataset
        idx = np.random.permutation(data.shape[0])

        if isinstance(data, torch.Tensor):
            data = data.numpy()  # to work with `ToPILImage'

        self.data = data[idx]

        # when STL10 'unlabelled'
        if not labels is None:
            self.labels = labels[idx]
        else:
            self.labels = labels

        self.transform = transform
        self.target_transform = target_transform
        self.two_crop = two_crop

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        # If the input data is in form from torchvision.datasets.ImageFolder
        if isinstance(self.data[index][0], np.str_):
            # Load image from path
            image = Image.open(self.data[index][0]).convert('RGB')

        else:
            # Get image / numpy pixel values
            image = self.data[index]

        if self.transform is not None:

            # Data augmentation and normalisation
            img = self.transform(image)

        if self.target_transform is not None:

            # Transforms the target, i.e. object detection, segmentation
            target = self.target_transform(target)

        if self.two_crop:

            # Augments the images again to create a second view of the data
            img2 = self.transform(image)

            #change
            # img3 = self.transform(image)

            # Combine the views to pass to the model    dim=0，按行拼接，dim=1，按列拼接
            img = torch.cat([img, img2], dim=0)

            # img = torch.cat([img, img2, img3], dim=0)

        # when STL10 'unlabelled'
        if self.labels is None:
            return img, torch.Tensor([0])
        else:
            return img, self.labels[index].long()


def random_split_image_folder(data, labels, n_classes, n_samples_per_class):
    """ Creates a class-balanced validation set from a training set.

        Specifically for the image folder class
    """

    train_x, train_y, valid_x, valid_y = [], [], [], []

    if isinstance(labels, list):
        labels = np.array(labels)

    for i in range(n_classes):
        # get indices of all class 'c' samples
        c_idx = (np.array(labels) == i).nonzero()[0]
        # get n unique class 'c' samples
        valid_samples = np.random.choice(c_idx, n_samples_per_class[i], replace=False)
        # get remaining samples of class 'c'
        # train_samples = np.setdiff1d(c_idx, valid_samples)
        # assign class c samples to validation, and remaining to training
        # train_x.extend(data[train_samples])
        # train_y.extend(labels[train_samples])
        train_x.extend(data[c_idx])
        train_y.extend(labels[c_idx])
        valid_x.extend(data[valid_samples])
        valid_y.extend(labels[valid_samples])

    # torch.from_numpy(np.stack(labels)) this takes the list of class ids and turns them to tensor.long

    return {'train': train_x, 'valid': valid_x}, \
        {'train': torch.from_numpy(np.stack(train_y)), 'valid': torch.from_numpy(np.stack(valid_y))}


def random_split(data, labels, n_classes, n_samples_per_class):
    """ Creates a class-balanced validation set from a training set.
    """

    train_x, train_y, valid_x, valid_y = [], [], [], []

    if isinstance(labels, list):        #isinstance判断是否为已知类型
        labels = np.array(labels)

    for i in range(n_classes):
        # get indices of all class 'c' samples
        c_idx = (np.array(labels) == i).nonzero()[0]
        # get n unique class 'c' samples
        valid_samples = np.random.choice(c_idx, n_samples_per_class[i], replace=False)
        # get remaining samples of class 'c'
        train_samples = np.setdiff1d(c_idx, valid_samples)     #找到2个数组中集合元素的差异
        # assign class c samples to validation, and remaining to training
        # train_x.extend(data[train_samples])
        # train_y.extend(labels[train_samples])
        train_x.extend(data[c_idx])                        # 不同之处？？？？？？？？？？？？？
        train_y.extend(labels[c_idx])
        valid_x.extend(data[valid_samples])
        valid_y.extend(labels[valid_samples])

    if isinstance(data, torch.Tensor):
        # torch.stack transforms list of tensors to tensor
        return {'train': torch.stack(train_x), 'valid': torch.stack(valid_x)}, \
            {'train': torch.stack(train_y), 'valid': torch.stack(valid_y)}
    # transforms list of np arrays to tensor
    return {'train': torch.from_numpy(np.stack(train_x)),
            'valid': torch.from_numpy(np.stack(valid_x))}, \
        {'train': torch.from_numpy(np.stack(train_y)),
         'valid': torch.from_numpy(np.stack(valid_y))}

def random_split_imagenet100(data, labels, pre_classes, n_classes, n_samples_per_class):
    """ Creates a class-balanced validation set from a training set.
    """

    pretrain_x, pretrain_y, train_x, train_y, valid_x, valid_y = [], [], [], [], [], []

    if isinstance(labels, list):
        labels = np.array(labels)

    max_classes = max(pre_classes, n_classes)

    for i in range(max_classes):
        # get indices of all class 'c' samples
        c_idx = (np.array(labels) == i).nonzero()[0]
        # get n unique class 'c' samples
        valid_samples = np.random.choice(c_idx, n_samples_per_class[i], replace=False)
        # get remaining samples of class 'c'
        train_samples = np.setdiff1d(c_idx, valid_samples)
        # assign class c samples to validation, and remaining to training
        if i < n_classes:
            train_x.extend(data[train_samples])
            train_y.extend(labels[train_samples])
            valid_x.extend(data[valid_samples])
            valid_y.extend(labels[valid_samples])
        if i < pre_classes:
            pretrain_x.extend(data[train_samples])
            pretrain_y.extend(labels[train_samples])

    if isinstance(data, torch.Tensor):
        # torch.stack transforms list of tensors to tensor
        return {'pretrain': pretrain_x, 'train': train_x, 'valid': valid_x}, \
            {'pretrain': torch.stack(pretrain_y), 'train': torch.stack(train_y), 'valid': torch.stack(valid_y)}
    # transforms list of np arrays to tensor
    return {'pretrain': pretrain_x,
            'train': train_x,
            'valid': valid_x}, \
        {'pretrain': torch.from_numpy(np.stack(pretrain_y)),
         'train': torch.from_numpy(np.stack(train_y)),
         'valid': torch.from_numpy(np.stack(valid_y))}


def sample_weights(labels):
    """ Calculates per sample weights. """
    class_sample_count = np.unique(labels, return_counts=True)[1]     #返回np中不重复的数字，并对每个数字进行计数
    class_weights = 1. / torch.Tensor(class_sample_count)
    return class_weights[list(map(int, labels))]


def experiment_config(parser, args):                        # 一些自己没设的参数，在这里了
    """ Handles experiment configuration and creates new dirs for model.
    """
    # check number of models already saved in 'experiments' dir, add 1 to get new model number
    run_dir = os.path.join(os.path.split(os.getcwd())[0], 'experiments')
    run_dir = '/mnt/data/donghengkui/experiments/simclr'

    os.makedirs(run_dir, exist_ok=True)

    run_name = time.strftime("%Y-%m-%d_%H-%M-%S")

    # create all save dirs
    model_dir = os.path.join(run_dir, run_name)

    os.makedirs(model_dir, exist_ok=True)

    args.summaries_dir = os.path.join(model_dir, 'summaries')
    args.checkpoint_dir = os.path.join(model_dir, 'checkpoint.pt')            # 需要修改一下，/mnt/data/donghengkui

    if not args.finetune:
        args.load_checkpoint_dir = args.checkpoint_dir

    os.makedirs(args.summaries_dir, exist_ok=True)

    # save hyperparameters in .txt file
    with open(os.path.join(model_dir, 'hyperparams.txt'), 'w') as logs:
        for key, value in vars(args).items():
            logs.write('--{0}={1} \n'.format(str(key), str(value)))

    # save config file used in .txt file
    with open(os.path.join(model_dir, 'config.txt'), 'w') as logs:
        # Remove the string from the blur_sigma value list
        config = parser.format_values().replace("'", "")
        # Remove the first line, path to original config file
        config = config[config.find('\n')+1:]
        logs.write('{}'.format(config))

    # reset root logger
    [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]
    # info logger for saving command line outputs during training
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=[logging.FileHandler(os.path.join(model_dir, 'trainlogs.txt')),
                                  logging.StreamHandler()])
    return args


def print_network(model, args):
    """ Utility for printing out a model's architecture.
    """
    logging.info('-'*70)  # print some info on architecture
    logging.info('{:>25} {:>27} {:>15}'.format('Layer.Parameter', 'Shape', 'Param#'))
    logging.info('-'*70)

    for param in model.state_dict():
        p_name = param.split('.')[-2]+'.'+param.split('.')[-1]
        # don't print batch norm layers for prettyness
        if p_name[:2] != 'BN' and p_name[:2] != 'bn':
            logging.info(
                '{:>25} {:>27} {:>15}'.format(
                    p_name,
                    str(list(model.state_dict()[param].squeeze().size())),
                    '{0:,}'.format(np.product(list(model.state_dict()[param].size())))
                )
            )
    logging.info('-'*70)

    logging.info('\nTotal params: {:,}\n\nSummaries dir: {}\n'.format(
        sum(p.numel() for p in model.parameters()),
        args.summaries_dir))

    for key, value in vars(args).items():
        if str(key) != 'print_progress':
            logging.info('--{0}: {1}'.format(str(key), str(value)))

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class CustomDataset_1(Dataset):
    """ Creates a custom pytorch dataset.

        - Creates two views of the same input used for unsupervised visual
        representational learning. (SimCLR, Moco, MocoV2)

    Args:
        data (array): Array / List of datasamples

        labels (array): Array / List of labels corresponding to the datasamples

        transforms (Dictionary, optional): The torchvision transformations
            to make to the datasamples. (Default: None)

        target_transform (Dictionary, optional): The torchvision transformations
            to make to the labels. (Default: None)

        two_crop (bool, optional): Whether to perform and return two views
            of the data input. (Default: False)

    Returns:
        img (Tensor): Datasamples to feed to the model.

        labels (Tensor): Corresponding lables to the datasamples.
    """

    def __init__(self, data, labels, transform=None, target_transform=None, two_crop=False):

        # shuffle the dataset
        idx = np.random.permutation(data.shape[0])

        if isinstance(data, torch.Tensor):
            data = data.numpy()  # to work with `ToPILImage'

        self.data = data[idx]

        # when STL10 'unlabelled'
        if not labels is None:
            self.labels = labels[idx]
        else:
            self.labels = labels

        self.transform = transform
        self.target_transform = target_transform
        self.two_crop = two_crop

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        # If the input data is in form from torchvision.datasets.ImageFolder
        if isinstance(self.data[index][0], np.str_):
            # Load image from path
            image = Image.open(self.data[index][0]).convert('RGB')
            #image = self.data[index]
            #image = image.transpose(1,2,0)

        else:
            # Get image / numpy pixel values
            image = self.data[index]
            image = image.transpose(1, 2, 0)

        if self.transform is not None:

            # Data augmentation and normalisation
            img = self.transform(image)

        if self.target_transform is not None:

            # Transforms the target, i.e. object detection, segmentation
            target = self.target_transform(target)

        if self.two_crop:

            # Augments the images again to create a second view of the data
            img2 = self.transform(image)

            #change
            # img3 = self.transform(image)

            # Combine the views to pass to the model    dim=0，按行拼接，dim=1，按列拼接
            img = torch.cat([img, img2], dim=0)

            # img = torch.cat([img, img2, img3], dim=0)

        # when STL10 'unlabelled'
        if self.labels is None:
            return img, torch.Tensor([0])
        else:
            return img, self.labels[index].long()


def tsne(encoder, dataloaders, mode,  args):
    ''' Evaluate script - MoCo
        Evaluate the encoder and Linear Evaluation head with Cross Entropy loss.
    '''

    epoch_valid_loss = None  # reset loss
    epoch_valid_acc = None  # reset acc
    epoch_valid_acc_top5 = None

    ''' Loss / Criterion '''
    criterion = nn.CrossEntropyLoss().cuda()

    # initilize Variables
    #args.writer = SummaryWriter(args.summaries_dir)

    # Evaluate both encoder and class head
    encoder.eval()
    # initilize Variables
    sample_count = 0
    run_loss = 0
    run_top1 = 0.0
    run_top5 = 0.0

    # Print setup for distributed only printing on one node.
    if args.print_progress:
            # tqdm for process (rank) 0 only when using distributed training
        eval_dataloader = tqdm(dataloaders[mode])
    else:
        eval_dataloader = dataloaders[mode]

    ''' epoch loop '''
    with torch.no_grad():
        for i, (inputs, target) in enumerate(eval_dataloader):

            # Do not compute gradient for encoder and classification head
            encoder.zero_grad()
            inputs = inputs.cuda()
            target = target.cuda()
            # Forward pass
            output = encoder(inputs)

            if i == 0:
                feature_bank = output
                label_bank = target
            else:
                feature_bank = torch.cat((feature_bank, output))
                label_bank = torch.cat((label_bank, target))

    feature_bank = feature_bank.cpu().numpy()
    label_bank = label_bank.cpu().numpy()
        #p, pseu = torch.max(torch.softmax(logits_bank, dim=-1), dim=-1)
        #prob_bank = p.cpu().numpy()
    tsne = TSNE(2)
    output = tsne.fit_transform(feature_bank)  # feature进行降维，降维至2维表示
        # 带真实值类别
    plt.title("1022040916donghengkui of python visualization", fontsize=15, loc='center', color='black')
    for i in range(10):  # 对每类的数据画上特定颜色的点
            index = (label_bank == i)
            plt.scatter(output[index, 0], output[index, 1], s=5, cmap=plt.cm.Spectral)
    plt.axis("off")
    plt.figure(dpi=400)
    #plt.legend(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    # plt.title("1022040916donghengkui of python visualization", fontsize=15, loc='center', color='black')
    plt.show()

def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels

def online_test(net, memory_data_loader, test_data_loader,  args):
    net.eval()
    classes = args.n_classes
    total_top1, total_top5, total_num, feature_bank, target_bank = 0.0, 0.0, 0, [], []
    with torch.no_grad():
        # generate feature bank
        for i, (data, target) in enumerate(memory_data_loader):
        # for data, target in enumerate(memory_data_loader):
            data = data.cuda()
            target = target.cuda()

            feature,_ = net(data)
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            target_bank.append(target)
        # [D, N]: D代表每个图像的特征维数, N代表dataset的大小
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        feature_labels = torch.cat(target_bank, dim=0).t().contiguous()
        # [N]
        for i, (data, target) in enumerate(test_data_loader):
        # for data, target in enumerate(memory_data_loader):
            data = data.cuda()
            target = target.cuda()

            feature = net(data)
            feature = F.normalize(feature, dim=1)
            # same with moco
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, 200, 0.1)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()

            # test_bar.set_description(
            #     'Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, args.epochs, total_top1 / total_num * 100))
            # print(
            #     'Epoch:{} * 200KNN-Acc@1 {top1_acc:.3f} 200KNN-Best_Acc@1 {best_acc:.3f} '.format(
            #         epoch, top1_acc=total_top1 / total_num * 100,
            #         best_acc=best_acc))

    return total_top1 / total_num * 100
