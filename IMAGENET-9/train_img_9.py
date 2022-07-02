'''Train ImageNet-9 with PyTorch.'''
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import models
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import csv
import argparse
import json
import time
from torch.autograd import Variable
from apex import amp
from multiprocessing import Pool

from torch.utils.data.distributed import DistributedSampler

from utils import progress_bar

import random
from PIL import Image,ImageFilter

import mixup as mp
import cutmix as cx
import mixup_v2 as mp_v2
import mixup_v3 as mp_v3
from lib.mixup_parallel import MixupProcessParallel
from lib.utils import *
from lib.validation import validate

import matplotlib.pyplot as plt
from dataset_img_9 import get_imagenet_dataloader

from puzzlemix.mixup_puzzle import mixup_graph

best_acc = 0
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
# # try:
# torch.multiprocessing.set_start_method("spawn")
#    print("spawned")
# except RuntimeError:
#    pass
def imshow(inp, title, ylabel):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()
    plt.ylabel('GroundTruth: {}'.format(ylabel))
    plt.title('predicted: {}'.format(title))



def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def train_comix(rank, mpp: MixupProcessParallel, print, configs, criterion, criterion_batch, train_loader,
          model, optimizer, epoch, lr_schedule,device,args):
    mean = torch.Tensor(np.array(configs.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3, configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(configs.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, configs.DATA.crop_size, configs.DATA.crop_size).cuda()

    # Initialize the meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()

    param_list = {
        'mixup_alpha': configs.TRAIN.alpha,
        'set_resolve': configs.TRAIN.set_resolve,
        'thres': configs.TRAIN.thres,
        'm_block_num': configs.TRAIN.block_num,
        'lam_dist': configs.TRAIN.lam_dist,
        'm_beta': configs.TRAIN.m_beta
    }

    #pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (input, target) in enumerate(train_loader):
        if (i == len(train_loader) -1 ):
            continue
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        # input, targets = input.to(device), targets.to(device)
        data_time.update(time.time() - end)

        # update learning rate
        lr = lr_schedule
        # lr_schedule(epoch + (i + 1) / len(train_loader))
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr

        optimizer.zero_grad()

        input.sub_(mean).div_(std)
        input_var = Variable(input, requires_grad=True)

        if configs.TRAIN.clean_lam == 0:
            model.eval()
        output = model(input_var)

        if configs.TRAIN.clean_lam > 0:
            loss_clean = configs.TRAIN.clean_lam * criterion(output, target)
        else:
            loss_clean = criterion(output, target)

        with amp.scale_loss(loss_clean, optimizer) as scaled_loss:
            scaled_loss.backward()

        unary = torch.sqrt(torch.mean(input_var.grad**2, dim=1))

        if configs.TRAIN.clean_lam == 0:
            model.train()

        # input = input.detach().cpu()
        target_reweighted = to_onehot(target, 9)

        # Calculating the distance between most salient regions
        with torch.no_grad():
            z = F.avg_pool2d(unary, kernel_size=8)
            z_reshape = z.reshape(args.batch_size, -1)
            z_idx_1d = torch.argmax(z_reshape, dim=1)
            z_idx_2d = torch.zeros(args.batch_size, 2)
            z_idx_2d[:, 0] = z_idx_1d // z.shape[-1]
            z_idx_2d[:, 1] = z_idx_1d % z.shape[-1]
            A_dist = distance(z_idx_2d, dist_type='l1').cuda()
            # print(A_dist,'A_dist')
            # parallel
            input, target_reweighted = mpp(input, target_reweighted, param_list, unary, A_dist)
        output = model(input)
        loss = torch.mean(torch.sum(-target_reweighted * nn.LogSoftmax(-1)(output), dim=1))

        # compute gradient and do SGD step
        if configs.TRAIN.clean_lam == 0:
            optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        # ----------------------------  ---------------------------- #
        # prec1, prec5 = accuracy(output, target, topk=(1, 5))
        # losses.update(loss.item(), input.size(0))
        # top1.update(prec1[0], input.size(0))
        # top5.update(prec5[0], input.size(0))

        # # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()

        progress_bar(i, len(train_loader)   )

def train_puzzlemix(train_loader, model, criterion, criterion_batch, optimizer, epoch, mean, std, lr_schedule,total_epoch,mp=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    current_LR = lr_schedule
    for i, (input, target) in enumerate(train_loader):
        optimizer.zero_grad()
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        r = np.random.rand(1)
        # calculate saliency map
        input_var = Variable(input, requires_grad=True)
        # if args.clean_lam == 0:
        model.eval()
        output = model(input_var)
        loss_clean = criterion(output, target)
        loss_clean.backward(retain_graph=False)
        optimizer.zero_grad()
        model.train()
        # else:
        #     # gradient regularization
        #     output = model(input_var)
        #     loss_clean = args.clean_lam * criterion(output, target)
        #     loss_clean.backward(retain_graph=True)

        unary = torch.sqrt(torch.mean(input_var.grad**2, dim=1))

        # perform mixup
        alpha = np.random.beta(1, 1)
        rand_index = torch.randperm(input.size()[0]).cuda()
        block_num = 2**np.random.randint(1, 5)
        with torch.no_grad():
            input, lam = mixup_graph(input,
                                        unary,
                                        rand_index,
                                        block_num=block_num,
                                        alpha=alpha,
                                        beta=1.2,
                                        gamma=0.5,
                                        eta=0.2,
                                        neigh_size=2,
                                        n_labels=3,
                                        mean=mean,
                                        std=std,
                                        transport=True,
                                        t_eps=0.8,
                                        dataset='imagenet',
                                        mp=mp)
        # calculate loss
        output = model(input)
        loss = lam * criterion_batch(output, target) + (1 - lam) * criterion_batch(
            output, target[rand_index])
        loss = torch.mean(loss)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        progress_bar(i, len(train_loader)   )

    return losses.avg

        
def test(epoch,net,testloader,criterion,device,total_epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    # i = 1
    # j = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # out = torchvision.utils.make_grid(inputs.data.cpu())
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            #print(np.array(classes)[predicted.data.cpu().numpy()])
            #print(np.array(classes)[targets.data.cpu().numpy()])
            #print(targets.data.cpu().numpy())
            
            # print(i,'.Predicted:', ''.join('%5s' % np.array(classes)[predicted.data.cpu().numpy()]),'  GroundTruth:',''.join('%5s' % np.array(classes)[targets.data.cpu().numpy()]))
            # if j % 4 == 0:
            #     plt.figure()
            #     j = j % 4
            # plt.subplot(2, 2, j + 1)
            # imshow(out, title=[np.array(classes)[predicted.data.cpu().numpy()]], ylabel=[np.array(classes)[targets.data.cpu().numpy()]])
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            progress_bar(batch_idx, len(testloader),
                        'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss / (batch_idx + 1), 100. * correct / total,
                            correct, total))

    # Save checkpoint.

    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            # 'net': net.state_dict(),
            'net': net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint/resnest50/'):
            os.mkdir('checkpoint/resnest50/')

        torch.save(state, './checkpoint/resnest50/ckpt.pth_' + 'resnest50' + '_epoch200_' + 'comix')
        best_acc = acc
    if epoch == total_epoch - 1:
        print("last model saving")
        state = {
            # 'net': net.state_dict(),
            'net': net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint/resnest50/'):
            os.mkdir('checkpoint/resnest50/')

        torch.save(state, './checkpoint/resnest50/ckpt.pth_' + 'resnest50' + '_' + 'comix' + '_' + 'last_model.pth')
    return (test_loss / batch_idx, 100. * correct / total)


# Training
def train(epoch,net,trainloader,criterion,device,args,optimizer):
    if args.local_rank == 0:  #仅显示第一个进程的训练过程，都显示的话会非常混乱
        print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        #print(inputs.shape)
        if args.mixup == 'ori':
            inputs, targets_a, targets_b, lam = mp.mixup_data(inputs, targets,
                                                              args.alpha)
            inputs = inputs.float()
            outputs = net(inputs)
            loss = mp.mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            train_loss += loss.item()
            _, predicted = outputs.max(1)

        elif args.mixup == 'cutmix':
            # inputs, lam = cx.mixup_data(inputs, targets, args.alpha)
            # inputs = inputs.float()
            # outputs = net(inputs)
            # loss = cx.mixup_criterion(criterion, outputs, targets, lam)
            # train_loss += loss.item()
            # _, predicted = outputs.max(1)
            r = np.random.rand(1)
            if args.beta > 0 and r < args.cutmix_prob:
                # generate mixed sample
                lam = np.random.beta(args.beta, args.beta)
                rand_index = torch.randperm(inputs.size()[0]).cuda()
                target_a = targets
                target_b = targets[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)

                inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))

                outputs = net(inputs)
                loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
            else:
                outputs = net(inputs)
                loss = criterion(outputs, targets)

            train_loss += loss.item()
            _, predicted = outputs.max(1)

        elif args.mixup == 'matrix':
            batch_size = inputs.size()[0]
            one_third = int(batch_size / 3)
            inputs_v2, targets_a_v2, targets_b_v2, lam_v2 = mp_v2.mixup_data(inputs, targets, args.alpha)
            inputs_v1, targets_a_v1, targets_b_v1, lam_v1 = mp.mixup_data(inputs, targets, args.alpha)

            inputs_mix = torch.cat((inputs[:one_third], inputs_v1[one_third:2 * one_third], inputs_v2[2 * one_third:]), 0)
            inputs_mix = inputs_mix.float()
            inputs_v2 = inputs_v2.float()
            inputs_v1 = inputs_v1.float()
            inputs_or = inputs.float()

            outputs_mix = net(inputs_mix)
            outputs_or = outputs_mix[:one_third]

            outputs_v1 = outputs_mix[one_third:2 * one_third]
            outputs_v2 = outputs_mix[2 * one_third:]

            loss_or = criterion(outputs_or, targets[:one_third])
            loss_v1 = mp.mixup_criterion(criterion, outputs_v1, targets_a_v1[one_third:2 * one_third], targets_b_v1[one_third:2 * one_third], lam_v1)
            loss_v2 = mp.mixup_criterion(criterion, outputs_v2, targets_a_v2[2 * one_third:], targets_b_v2[2 * one_third:], lam_v2)

            loss = (loss_v2 + loss_or + loss_v1) / 3
            train_loss += loss.item()
            _, predicted = torch.max(outputs_v2.data, 1)

        elif args.mixup == 'one_fourth':
            batch_size = inputs.size()[0]
            one_fourth = int(batch_size / 4)
            inputs_v1, targets_a_v1, targets_b_v1, lam_v1 = mp.mixup_data(inputs, targets, args.alpha)
            inputs_v2, targets_a_v2, targets_b_v2, lam_v2 = mp_v2.mixup_data(inputs, targets, args.alpha)
            inputs_v3, targets_a_v3, targets_b_v3, lam_v3 = mp_v3.mixup_data(inputs, targets, args.slice_num)
            inputs_mix = torch.cat((inputs[:one_fourth], inputs_v1[one_fourth:2 * one_fourth],
                                    inputs_v2[2 * one_fourth: 3 * one_fourth], inputs_v3[3 * one_fourth:]), 0)
            inputs_mix = inputs_mix.float()
            inputs_or = inputs.float()
            inputs_v1 = inputs_v1.float()
            inputs_v2 = inputs_v2.float()
            inputs_v3 = inputs_v3.float()

            outputs_mix = net(inputs_mix)
            outputs_or = outputs_mix[:one_fourth]
            outputs_v1 = outputs_mix[one_fourth:2 * one_fourth]
            outputs_v2 = outputs_mix[2 * one_fourth:3 * one_fourth]
            outputs_v3 = outputs_mix[3 * one_fourth:]

            loss_or = criterion(outputs_or, targets[:one_fourth])
            loss_v1 = mp.mixup_criterion(criterion, outputs_v1, targets_a_v1[one_fourth:2 * one_fourth],
                                         targets_b_v1[one_fourth:2 * one_fourth], lam_v1)
            loss_v2 = mp.mixup_criterion(criterion, outputs_v2, targets_a_v2[2 * one_fourth:3 * one_fourth],
                                         targets_b_v2[2 * one_fourth:3 * one_fourth], lam_v2)
            loss_v3 = mp.mixup_criterion(criterion, outputs_v3, targets_a_v3[3 * one_fourth:],
                                         targets_b_v3[3 * one_fourth:], lam_v3)
            loss = (loss_v3 + loss_v2 + loss_or + loss_v1) / 4
            train_loss += loss.item()
            _, predicted = torch.max(outputs_v3.data, 1)

        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            train_loss += loss.item()
            _, predicted = outputs.max(1)

        # train_loss += loss.item()
        # _, predicted = outputs.max(1)
        total += targets.size(0)
        correct = 0.0
        # correct += predicted.eq(targets).sum().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: idia-%.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if args.local_rank == 0:  #仅显示第一个进程的训练过程，都显示的话会非常混乱
            progress_bar(batch_idx, len(trainloader),
                        'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                        % (train_loss / (batch_idx + 1), reg_loss / (batch_idx + 1),
                            100. * correct / total, correct, total))

    return (train_loss / batch_idx, reg_loss / batch_idx, 100. * correct / total)

def main(self):

    parser = argparse.ArgumentParser(description='PyTorch ImageNet-9 Training')
    parser.add_argument('--lr', default=0.025, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--name', default='0', type=str, help='name of run')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--model', default="resnest101", type=str,
                        help='model type (default: resnest50)')
    parser.add_argument('--alpha', default=1., type=float,
                        help='mixup interpolation coefficient (default: 1)')
    parser.add_argument('--mixup', type=str, default='ori', help='mixup method')
    parser.add_argument('--epoch', default=200, type=int,
                        help='total epochs to run')

    parser.add_argument('--input_size', default=224, type=int,
                        help='the size of input image')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='total epochs to run')                   
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--beta', default=0, type=float,
                        help='hyperparameter beta')
    parser.add_argument('--cutmix_prob', default=0, type=float,
                        help='cutmix probability')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')

    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')

    parser.add_argument('--slice_num', default=3, type=int,
                        help='number of image slice in mixup_v3')
    parser.add_argument('-c',
                            '--config',
                            default='configs/comix/configs_fast_phase2.yml',
                            type=str,
                            metavar='Path',
                            help='path to the config file (default: configs.yml)')
    parser.add_argument('--output_prefix',
                            default='fast_adv',
                            type=str,
                            help='prefix used to define output path')
    args = parser.parse_args()

    configs = parse_config_file(args)
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')

    # # Model
    print('==> Building model..')


    # 使用DDP进行多进程分布式训练，local_rank表示系统自动分配的进程号，
    torch.cuda.set_device(args.local_rank)  # 这里设定每一个进程使用的GPU是一定的，即一张gpu一个进程
    device = torch.device('cuda', args.local_rank)
    # torch.distributed.init_process_group(backend='nccl') #初始化"gloo"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "11111"
    torch.distributed.init_process_group(backend="gloo") #初始化"gloo"

    # 固定随机种子，使用相同的参数进行不同进程上模型的初始化
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    net = models.__dict__[args.model]()

    def init_dist_weights(net):
            for m in net.modules():
                # if isinstance(m, BasicBlock):
                #     m.bn2.weight = nn.Parameter(torch.zeros_like(m.bn2.weight))
                if isinstance(m, models.resnest.Bottleneck):
                    m.bn3.weight = nn.Parameter(torch.zeros_like(m.bn3.weight))
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
    if args.mixup == 'comix':
        init_dist_weights(net)
    net = net.to(device)

    transform_train = transforms.Compose([
        transforms.Resize([args.input_size,args.input_size]),
        transforms.RandomCrop([args.input_size,args.input_size], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([args.input_size,args.input_size]),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    batch_size = args.batch_size
    train_dir='./imagenet-1k/train'
    test_dir = './imagenet-1k/val'


    trainset =  get_imagenet_dataloader(train_dir, batch_size=batch_size, transform = transform_train,train=True, val_data='ImageNet-A',)
    train_sampler = DistributedSampler(trainset)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, num_workers=1,sampler=train_sampler) # 这个sampler会自动分配数据到各个gpu上

    testset = get_imagenet_dataloader(test_dir, batch_size=batch_size,transform = transform_test, train=False, val_data='ImageNet-A',)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size,  num_workers=1,sampler=DistributedSampler(testset,shuffle=False))

    criterion = nn.CrossEntropyLoss()
    criterion_batch = nn.CrossEntropyLoss(reduction='none').cuda()
    # optimizer = optim.SGD(net.parameters(), lr=args.lr,
    #                       momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.SGD(net.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    mean_torch = torch.tensor(mean, dtype=torch.float32).reshape(1, 3, 1, 1).cuda()
    std_torch = torch.tensor(std, dtype=torch.float32).reshape(1, 3, 1, 1).cuda()
    

    if args.mixup == 'comix':
        net, optimizer = amp.initialize(net, optimizer, opt_level="O1", loss_scale=1024)
    # print(len(testset))
    # print(len(testloader))
    classes = testset.classes
    # print(classes)
    net = torch.nn.parallel.DistributedDataParallel(net,
                                            device_ids=[args.local_rank], 
                                            output_device=args.local_rank,
                                            find_unused_parameters=True) #将模型分布到多张gpu上

    cudnn.benchmark = True

    if not os.path.isdir('results'):
        os.mkdir('results')
    logname = ('results/log' +  '_' + args.model + '_epoch200_' + args.mixup
            + str(args.seed) + '.csv')

    if args.mixup == 'comix':    
        mpp = MixupProcessParallel(part=16, num_thread=2)


    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
                                'test loss', 'test acc'])


    for epoch in range(start_epoch, args.epoch):
        # train_sampler.set_epoch(epoch)
        if args.mixup == 'comix':
            train_comix(args.local_rank, mpp, print, configs, criterion, criterion_batch, trainloader, net, optimizer,
                    epoch, scheduler,device,args)
        elif args.mixup == 'puzzlemix':
            train_loss = train_puzzlemix(trainloader, net, criterion, criterion_batch, optimizer, epoch, mean_torch, std_torch,
                    scheduler,args.epoch, mp= Pool(1))
        else:
            train_loss, reg_loss, train_acc = train(epoch,net,trainloader,criterion,device,args,optimizer)
        # train_loss, reg_loss, train_acc = train_comix(epoch)
        test_loss, test_acc = test(epoch,net,testloader,criterion,device,args.epoch)
        with open(logname, 'a') as logfile:
            if args.local_rank == 0: #仅记录第一个进程的测试结果，由于参数共享，不同进程测试结果相同，无需重复保存
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow([epoch, test_loss, test_acc])
            scheduler.step()
    print("end epoch")
    mpp.close()
    print("mpp close")
    torch.distributed.destroy_process_group()
    print("end cleanup")


if __name__ == '__main__':
    torch.multiprocessing.spawn(main)   #对主函数使用多线程，启动方式为spawn

