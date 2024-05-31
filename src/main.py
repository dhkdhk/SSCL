

import os
#os.environ['CUDA_VISIBLE_DEVICES']='0'            # 指定GPU1
#os.environ['CUDA_VISIBLE_DEVICES']='1'            # 指定GPU1
#os.environ['CUDA_VISIBLE_DEVICES']='2'            # 指定GPU2
import sys
sys.path.append('/home/donghengkui/simclr-dhk')

import ssl
ssl._create_default_https_context = ssl._create_unverified_context          # 处理未知的错误
import argparse
import configargparse
import warnings
import datetime
from torch.nn.parallel import DistributedDataParallel
import torch.distributed
from train import finetune, evaluate, pretrain, supervised
from datasets import get_dataloaders                   # 数据集加载
from utils import *                                    # 一些功能性函数
import model.network as models                         # resnet骨干
from model.SimCLR_SSCL import SimClr_SSCL



warnings.filterwarnings("ignore")
default_config = os.path.join(os.path.split(os.getcwd())[0], 'config.conf')         #默认配置
parser = configargparse.ArgumentParser(
     description='Pytorch MocoV2', default_config_files=[default_config])
parser.add_argument('-c', '--my-config', required=False,
                    is_config_file=True, help='config file path')
parser.add_argument('--dataset', default='cifar10',
                    help='Dataset, (Options: cifar10, cifar100, stl10, imagenet, tinyimagenet).')
parser.add_argument('--method', default='simclr',
                    help='method, (Options: simclr, simclr_hcl, simclr_mix, simclr_sscl')
parser.add_argument('--dataset_path', default=None,
                    help='Path to dataset, Not needed for TorchVision Datasets.')
parser.add_argument('--model', default='resnet18',
                    help='Model, (Options: resnet18, resnet34, resnet50, resnet101, resnet152).')
parser.add_argument('--n_epochs', type=int, default=200,
                    help='Number of Epochs in Contrastive Training.')
parser.add_argument('--finetune_epochs', type=int, default=100,
                    help='Number of Epochs in Linear Classification Training.')
parser.add_argument('--warmup_epochs', type=int, default=20,
                    help='Number of Warmup Epochs During Contrastive Training.')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of Samples Per Batch.')
parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='Starting Learing Rate for Contrastive Training.')
parser.add_argument('--base_lr', type=float, default=0.0001,
                    help='Base / Minimum Learing Rate to Begin Linear Warmup.')
parser.add_argument('--finetune_learning_rate', type=float, default=10.0,
                    help='Starting Learing Rate for Linear Classification Training.')
parser.add_argument('--weight_decay', type=float, default=1e-3,
                    help='Contrastive Learning Weight Decay Regularisation Factor.')
parser.add_argument('--finetune_weight_decay', type=float, default=0.0,
                    help='Linear Classification Training Weight Decay Regularisation Factor.')
parser.add_argument('--optimiser', default='sgd',
                    help='Optimiser, (Options: sgd, adam, lars).')
parser.add_argument('--patience', default=100, type=int,
                    help='Number of Epochs to Wait for Improvement.')
parser.add_argument('--queue_size', type=int, default=16384,
                    help='Size of Memory Queue, Must be Divisible by batch_size.')
parser.add_argument('--queue_momentum', type=float, default=0.99,
                    help='Momentum for the Key Encoder Update.')
parser.add_argument('--temperature', type=float, default=0.5,
                    help='InfoNCE Temperature Factor')

parser.add_argument('--tau_plus', default=0.1,type=float,
                    help='tau_plus')
parser.add_argument('--beta', default=1, type=float,
                    help='beta')
parser.add_argument('--estimator',  default='easy',
                    help='simclr/simclr_hcl')
parser.add_argument('--s', default=64,type=int,
                    help='s')
parser.add_argument('--k', default=32,type=int,
                    help='k')

parser.add_argument('--alpha', default=0.5, type=float,
                    help='alpha')


parser.add_argument('--a', default=0.5,type=float,
                    help='tau_plus')
parser.add_argument('--b', default=0, type=float,
                    help='b')
parser.add_argument('--cat',action='store_true',
                    help='Perform Only Linear Classification Training. (Default: False)')

parser.add_argument('--jitter_d', type=float, default=0.5,
                    help='Distortion Factor for the Random Colour Jitter Augmentation')
parser.add_argument('--jitter_p', type=float, default=0.8,
                    help='Probability to Apply Random Colour Jitter Augmentation')
parser.add_argument('--blur_sigma', nargs=2, type=float, default=[0.1, 2.0],
                    help='Radius to Apply Random Colour Jitter Augmentation')
parser.add_argument('--blur_p', type=float, default=0.5,
                    help='Probability to Apply Gaussian Blur Augmentation')
parser.add_argument('--grey_p', type=float, default=0.2,
                    help='Probability to Apply Random Grey Scale')
parser.add_argument('--no_twocrop', dest='twocrop', action='store_false',
                    help='Whether or Not to Use Two Crop Augmentation, Used to Create Two Views of the Input for Contrastive Learning. (Default: True)')
parser.set_defaults(twocrop=True)

parser.add_argument('--load_checkpoint_dir', default=r'./home/donghengkui/moco-dhk/experiments/checkpoint.pt',
                    help='Path to Load Pre-trained Model From.')
parser.add_argument('--load_checkpoint_dir_finetune', default=r'./home/donghengkui/moco-dhk/experiments/checkpoint.pt',
                    help='Path to Load Finetune Model From.')

parser.add_argument('--no_distributed', dest='distributed', action='store_false',
                    help='Whether or Not to Use Distributed Training. (Default: True)')
parser.set_defaults(distributed=False)

parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Perform Only Linear Classification Training. (Default: False)')
#parser.set_defaults(finetune=False)

parser.add_argument('--supervised', dest='supervised', action='store_true',
                    help='Perform Supervised Pre-Training. (Default: False)')
#parser.set_defaults(supervised=False)


def setup(distributed):
    if distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        local_rank = int(os.environ.get('LOCAL_RANK'))
        device = torch.device(f'cuda:{local_rank}')  # unique on individual node

        print('World size: {} ; Rank: {} ; LocalRank: {} ; Master: {}:{}'.format(
            os.environ.get('WORLD_SIZE'),
            os.environ.get('RANK'),
            os.environ.get('LOCAL_RANK'),
            os.environ.get('MASTER_ADDR'), os.environ.get('MASTER_PORT')))
    else:
        local_rank = None
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    seed = 333

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # True

    return device, local_rank


def main():
    """ Main """
    #time
    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Arguments
    args = parser.parse_args()
    # Setup Distributed Training
    device, local_rank = setup(distributed=args.distributed)
    # Get Dataloaders for Dataset of choice
    dataloaders, args = get_dataloaders(args)
    # Setup logging, saving models, summaries
    args = experiment_config(parser, args)

    ''' Base Encoder '''
    # Get available models from /model/network.py
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))
    # If model exists
    if any(args.model in model_name for model_name in model_names):
        # Load model
        base_encoder = getattr(models, args.model)(                      # base_encoder resnet18
            args, num_classes=args.n_classes)  # Encoder
    else:
        raise NotImplementedError("Model Not Implemented: {}".format(args.model))

    if not args.supervised:
        # freeze all layers but the last fc
        for name, param in base_encoder.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
        # init the fc layer
        init_weights(base_encoder)


    ''' SSCL Model '''

    simclr = SimClr_SSCL(args,device)
    tsne= True
    # simclr = SimClr_Model(args)


    # Place model onto GPU(s)
    if args.distributed:
        torch.cuda.set_device(device)
        torch.set_num_threads(6)  # n cpu threads / n processes per node

        simclr = DistributedDataParallel(simclr.cuda(),
                                       device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True, broadcast_buffers=False)
        base_encoder = DistributedDataParallel(base_encoder.cuda(),
                                               device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True, broadcast_buffers=False)

        # Only print from process (rank) 0
        args.print_progress = True if int(os.environ.get('RANK')) == 0 else False
    else:

        simclr.to(device)

        base_encoder.to(device)
        args.print_progress = True
    # launch model training or inference
    if not args.finetune:                          # finetune 默认为false
        ''' Pretraining / Finetuning / Evaluate '''
        if not args.supervised:
            # Pretrain the encoder and projection head
            #pretrain(moco_model_2network, dataloaders, args)

            pretrain(simclr, dataloaders, args, device)
            #pretrain(moco, dataloaders, args)
            #pretrain(mocov2, dataloaders, args)
            # Load the state_dict from query encoder and load it on finetune net
            base_encoder = load_moco(base_encoder, args)


        else:
            supervised(base_encoder, dataloaders, args)


            # Load the state_dict from query encoder and load it on finetune net
            base_encoder = load_sup(base_encoder, args)


        # Supervised Finetuning of the supervised classification head
        finetune(base_encoder, dataloaders, args)

        # Evaluate the pretrained model and trained supervised head
        test_loss, test_acc, test_acc_top5 = evaluate(
            base_encoder, dataloaders, 'test', args.finetune_epochs, args)

        print('[Test] loss {:.4f} - acc {:.4f} - acc_top5 {:.4f}'.format(
            test_loss, test_acc, test_acc_top5))
        end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logging.info('\ntime {}/{}:\n'.format(start_time, end_time))
        if args.distributed:  # cleanup
            torch.distributed.destroy_process_group()

    else:
        ''' Finetuning / Evaluate '''
        # Do not Pretrain, just finetune and inference
        # Load the state_dict from query encoder and load it on finetune net

        base_encoder = load_moco(base_encoder, args)
        # # Supervised Finetuning of the supervised classification head
        finetune(base_encoder, dataloaders, args)
        #base_encoder =load_finetune(base_encoder, args)

        # Evaluate the pretrained model and trained supervised head

        # base_encoder = load_finetune(base_encoder, args)
        test_loss, test_acc, test_acc_top5 = evaluate(
            base_encoder, dataloaders, 'test', args.finetune_epochs, args)
        print('[Test] loss {:.4f} - acc {:.4f} - acc_top5 {:.4f}'.format(
            test_loss, test_acc, test_acc_top5))
        if args.distributed:  # cleanup
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
