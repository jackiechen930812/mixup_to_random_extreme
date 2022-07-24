'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import argparse
import csv
import os

import numpy as np
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import mix_aug
import mixup as mp
import mixup_v2 as mp_v2
import mixup_v3 as mp_v3
import models
from comix.mixup import mixup_process
from comix.utils import to_one_hot, distance
from models import *
from puzzlemix.mixup import mixup_process as mixup_process_p
from puzzlemix.mixup import to_one_hot as to_one_hot_p
from utils import progress_bar, top_accuracy, calib_err


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--model', default="ResNet18", type=str,
                    help='model type (default: ResNet18)')  # WideResNet
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--mixup', type=str, default='ori', help='mixup method')
parser.add_argument('--epoch', default=200, type=int,
                    help='total epochs to run')
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

# comix
parser.add_argument('--m_block_num',
                    type=int,
                    default=4,
                    help='resolution of labeling, -1 for random')
parser.add_argument('--m_part', type=int, default=20, help='partition size')
parser.add_argument('--m_beta', type=float, default=0.32, help='label smoothness coef, 0.16~1.0')
parser.add_argument('--m_gamma', type=float, default=1.0, help='supermodular diversity coef')
parser.add_argument('--m_thres',
                    type=float,
                    default=0.83,
                    help='threshold for over-penalization, tau, 0.81~0.86')
parser.add_argument('--m_thres_type',
                    type=str,
                    default='hard',
                    choices=['soft', 'hard'],
                    help='thresholding type')
parser.add_argument('--m_eta', type=float, default=0.05, help='prior coef')
parser.add_argument('--mixup_alpha',
                    type=float,
                    default=2.0,
                    help='alpha parameter for dirichlet prior')
parser.add_argument('--m_omega', type=float, default=0.001, help='input compatibility coef, \omega')
parser.add_argument('--set_resolve',
                    # type=str2bool,
                    default=True,
                    help='post-processing for resolving the same outputs')
parser.add_argument('--m_niter', type=int, default=4, help='number of outer iteration')
parser.add_argument('--clean_lam', type=float, default=1.0, help='clean input regularization')

# puzzlemix
parser.add_argument('--box', type=str2bool, default=False, help='true for CutMix')
parser.add_argument('--graph', type=str2bool, default=True, help='true for PuzzleMix')
parser.add_argument('--neigh_size',
                    type=int,
                    default=4,
                    help='neighbor size for computing distance beteeen image regions')
parser.add_argument('--n_labels', type=int, default=3, help='label space size')
parser.add_argument('--transport', type=str2bool, default=True, help='whether to use transport')
parser.add_argument('--t_eps', type=float, default=0.8, help='transport cost coefficient')
parser.add_argument('--t_size',
                    type=int,
                    default=-1,
                    help='transport resolution. -1 for using the same resolution with graphcut')
parser.add_argument('--adv_eps', type=float, default=10.0, help='adversarial training ball')
parser.add_argument('--adv_p', type=float, default=0.0, help='adversarial training probability')
parser.add_argument('--mp', type=int, default=8, help='multi-process for graphcut (CPU)')
parser.add_argument('--in_batch',
                    type=str2bool,
                    default=False,
                    help='whether to use different lambdas in batch')

# training
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--severity', type=int, default=3)
parser.add_argument('--transfer_datas', type=bool, default=False)
# parser.add_argument('--learning_rate', type=float, default=0.2)
# parser.add_argument('--momentum', type=float, default=0.9)
# parser.add_argument('--decay', type=float, default=0.0001, help='weight decay (L2 penalty)')
# parser.add_argument('--schedule',
#                     type=int,
#                     nargs='+',
#                     default=[100, 200],
#                     help='decrease learning rate at these epochs')
parser.add_argument(
    '--gammas',
    type=float,
    nargs='+',
    default=[0.1, 0.1],
    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

stride = 1
args.mean = torch.tensor([x / 255 for x in [125.3, 123.0, 113.9]],
                         dtype=torch.float32).reshape(1, 3, 1, 1).cuda()
args.std = torch.tensor([x / 255 for x in [63.0, 62.1, 66.7]],
                        dtype=torch.float32).reshape(1, 3, 1, 1).cuda()
args.labels_per_class = 5000


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


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
if args.mixup=="AugMix":
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
preprocess = transform_test

if args.transfer_datas:
    trainset = torchvision.datasets.ImageFolder('./transfer_res/images', transform=transform_train)
else:
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
if args.mixup == 'AugMix':
    trainset = mix_aug.AugMixDataset(trainset, preprocess)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

train_features, train_labels = next(iter(trainloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat', 'deer',
#            'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = ResNet18()
# net = WideResNet(28, 10, 0.3)
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
# net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(
        './checkpoint/ResNet18/ckpt.pth' + '_' + args.model + '_' + str(args.epoch) + '_' + args.mixup + '_'
        + str(args.seed))
    # net.load_state_dict(checkpoint['net'])
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)
else:
    print('==> Building model..')
    net = models.__dict__[args.model]()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if not os.path.isdir('results'):
    os.mkdir('results')
logname = ('results/log' + '_' + args.model + '_' + str(args.epoch) + '_' + args.mixup + '_'
           + str(args.seed) + '.csv')

bce_loss = nn.BCELoss().cuda()  # 二进制交叉熵损失函数
bce_loss_sum = nn.BCELoss(reduction='sum').cuda()  # sum 所有样本的loss相加
softmax = nn.Softmax(dim=1).cuda()
criterion = nn.CrossEntropyLoss()
criterion_batch = nn.CrossEntropyLoss(reduction='none').cuda()  # none 每个样本产生一个loss，共batch_size个值
# optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                       momentum=0.9, weight_decay=5e-4)
optimizer = torch.optim.SGD(net.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    top1_acc, top5_acc = 0., 0.
    rms_confidence, rms_correct = [], []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        if args.mixup == 'ori':
            inputs, targets_a, targets_b, lam = mp.mixup_data(inputs, targets,
                                                              args.alpha)
            inputs = inputs.float()
            outputs = net(inputs)
            loss = mp.mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            train_loss += loss.item()
            _, predicted = outputs.max(1)

        elif args.mixup == 'AugMix':
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)
            train_loss += loss.item()
            _, predicted = outputs.max(1)

        elif args.mixup == 'matrix':
            inputs, targets_a, targets_b, lam = mp_v2.mixup_data(inputs, targets,
                                                                 args.alpha)
            inputs = inputs.float()
            outputs = net(inputs)
            loss = mp_v2.mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            train_loss += loss.item()
            _, predicted = outputs.max(1)

        elif args.mixup == 'one-second':
            batch_size = inputs.size()[0]
            one_second = int(batch_size / 2)
            inputs_v1, targets_a_v1, targets_b_v1, lam_v1 = mp.mixup_data(inputs, targets, args.alpha)
            inputs_v2, targets_a_v2, targets_b_v2, lam_v2 = mp_v2.mixup_data(inputs, targets, args.alpha)

            inputs_mix = torch.cat((inputs_v1[:one_second], inputs_v2[one_second:]), 0)
            inputs_mix = inputs_mix.float()
            inputs_v2 = inputs_v2.float()
            inputs_v1 = inputs_v1.float()

            outputs_mix = net(inputs_mix)
            outputs_v1 = outputs_mix[:one_second]
            outputs_v2 = outputs_mix[one_second:]
            loss_v1 = mp.mixup_criterion(criterion, outputs_v1, targets_a_v1[:one_second],
                                         targets_b_v1[:one_second], lam_v1)
            loss_v2 = mp.mixup_criterion(criterion, outputs_v2, targets_a_v2[one_second:], targets_b_v2[one_second:],
                                         lam_v2)
            loss = (loss_v2 + loss_v1) / 2
            train_loss += loss.item()
            _, predicted = torch.max(outputs_v2.data, 1)

        elif args.mixup == 'comix':
            input_var = Variable(inputs, requires_grad=True)
            target_var = Variable(targets)

            outputs = net(input_var)
            loss_batch = 2 * criterion_batch(outputs, target_var) / 10  # 此处10为类别数，如更换数据集需修改为对应类别
            loss_batch_mean = torch.mean(loss_batch, dim=0)
            loss_batch_mean.backward(retain_graph=True)
            sc = torch.sqrt(torch.mean(input_var.grad ** 2, dim=1))

            # Here, we calculate distance between most salient location (Compatibility)
            # We can try various measurements
            with torch.no_grad():
                z = F.avg_pool2d(sc, kernel_size=8, stride=1)
                z_reshape = z.reshape(args.batch_size, -1)
                z_idx_1d = torch.argmax(z_reshape, dim=1)
                z_idx_2d = torch.zeros((args.batch_size, 2), device=z.device)
                z_idx_2d[:, 0] = z_idx_1d // z.shape[-1]
                z_idx_2d[:, 1] = z_idx_1d % z.shape[-1]
                A_dist = distance(z_idx_2d, dist_type='l1')
            target_reweighted = to_one_hot(target_var, 10)
            out, target_reweighted = mixup_process(inputs,
                                                   target_reweighted,
                                                   args=args,
                                                   sc=sc,
                                                   A_dist=A_dist)
            out = net(out)
            loss = bce_loss(softmax(out), target_reweighted)
            train_loss += loss.item()
            _, predicted = out.max(1)

        elif args.mixup == 'puzzlemix':
            unary = None
            noise = None
            adv_mask1 = 0
            adv_mask2 = 0

            input_var = Variable(inputs, requires_grad=True)
            target_var = Variable(targets)
            output = net(input_var)
            loss_batch = 2 * criterion_batch(output, target_var) / 10  # change number of classes
            loss_batch_mean = torch.mean(loss_batch, dim=0)
            loss_batch_mean.backward(retain_graph=True)
            unary = torch.sqrt(torch.mean(input_var.grad ** 2, dim=1))

            # input_var, target_var = Variable(inputs), Variable(targets)
            target_reweighted = to_one_hot_p(targets, 10)  # change number of classes
            out, target_reweighted = mixup_process_p(inputs,
                                                     target_reweighted,
                                                     args=args,
                                                     grad=unary,
                                                     noise=noise,
                                                     adv_mask1=adv_mask1,
                                                     adv_mask2=adv_mask2)
            outputs = net(out)
            loss = bce_loss(softmax(outputs), target_reweighted)
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

        elif args.mixup == 'one-third':
            batch_size = inputs.size()[0]
            one_third = int(batch_size / 3)
            inputs_v2, targets_a_v2, targets_b_v2, lam_v2 = mp_v2.mixup_data(inputs, targets, args.alpha)
            inputs_v1, targets_a_v1, targets_b_v1, lam_v1 = mp.mixup_data(inputs, targets, args.alpha)

            inputs_mix = torch.cat((inputs[:one_third], inputs_v1[one_third:2 * one_third], inputs_v2[2 * one_third:]),
                                   0)
            inputs_mix = inputs_mix.float()
            inputs_v2 = inputs_v2.float()
            inputs_v1 = inputs_v1.float()
            inputs_or = inputs.float()

            outputs = net(inputs_mix)
            outputs_or = outputs[:one_third]

            outputs_v1 = outputs[one_third:2 * one_third]
            outputs_v2 = outputs[2 * one_third:]

            loss_or = criterion(outputs_or, targets[:one_third])
            loss_v1 = mp.mixup_criterion(criterion, outputs_v1, targets_a_v1[one_third:2 * one_third],
                                         targets_b_v1[one_third:2 * one_third], lam_v1)
            loss_v2 = mp.mixup_criterion(criterion, outputs_v2, targets_a_v2[2 * one_third:],
                                         targets_b_v2[2 * one_third:], lam_v2)

            loss = (loss_v2 + loss_or + loss_v1) / 3
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

        elif args.mixup == 'one-fourth':
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

            outputs = net(inputs_mix)
            outputs_or = outputs[:one_fourth]
            outputs_v1 = outputs[one_fourth:2 * one_fourth]
            outputs_v2 = outputs[2 * one_fourth:3 * one_fourth]
            outputs_v3 = outputs[3 * one_fourth:]

            loss_or = criterion(outputs_or, targets[:one_fourth])
            loss_v1 = mp.mixup_criterion(criterion, outputs_v1, targets_a_v1[one_fourth:2 * one_fourth],
                                         targets_b_v1[one_fourth:2 * one_fourth], lam_v1)
            loss_v2 = mp.mixup_criterion(criterion, outputs_v2, targets_a_v2[2 * one_fourth:3 * one_fourth],
                                         targets_b_v2[2 * one_fourth:3 * one_fourth], lam_v2)
            loss_v3 = mp.mixup_criterion(criterion, outputs_v3, targets_a_v3[3 * one_fourth:],
                                         targets_b_v3[3 * one_fourth:], lam_v3)

            loss = (loss_v3 + loss_v2 + loss_or + loss_v1) / 4
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

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

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        top1_acc_item, top5_acc_item = top_accuracy(outputs.detach().cpu(), targets.detach().cpu(), topk=(1,5))
        top1_acc += top1_acc_item
        top5_acc += top5_acc_item
        rms_confidence.extend(F.softmax(outputs.detach().cpu()).squeeze().tolist())
        rms_correct.extend(predicted.eq(targets).cpu().squeeze().tolist())
        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d) | Top1 Acc: %.3f | Top5 Acc: %.3f'
                     % (train_loss / (batch_idx + 1), reg_loss / (batch_idx + 1),
                        100. * correct / total, correct, total, 100. * top1_acc_item, 100. * top5_acc_item))
    train_rms = 100 * calib_err(rms_confidence, rms_correct, p='2', beta=10)
    return (train_loss / batch_idx, reg_loss / batch_idx, 100. * correct / total, 100. * top1_acc / len(trainloader), 100. * top5_acc / len(trainloader), train_rms)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    top1_acc, top5_acc = 0., 0.
    rms_confidence, rms_correct = [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # for precisely finding the wrong img, set batchsize as 1
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            top1_acc_item, top5_acc_item = top_accuracy(outputs.detach().cpu(), targets.detach().cpu(), topk=(1, 5))
            top1_acc += top1_acc_item
            top5_acc += top5_acc_item
            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            rms_confidence.extend(F.softmax(outputs.detach().cpu()).squeeze().tolist())
            rms_correct.extend(predicted.eq(targets).cpu().squeeze().tolist())
            progress_bar(batch_idx, len(testloader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d) | Top1 Acc: %.3f | Top5 Acc: %.3f'
                         % (test_loss / (batch_idx + 1), 100. * correct / total,
                            correct, total, 100. * top1_acc_item, 100. * top5_acc_item))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            # 'net': net.state_dict(),
            'net': net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint/ResNet18/'):
            os.mkdir('checkpoint/ResNet18/')

        torch.save(state,
                   './checkpoint/ResNet18/ckpt.pth' + '_' + args.model + '_' + str(args.epoch) + '_' + args.mixup + '_'
                   + str(args.seed) + '.pth')
        best_acc = acc

    if epoch == args.epoch - 1:
        print("last model saving")
        state = {
            # 'net': net.state_dict(),
            'net': net,
            'acc': acc,
            'epoch': epoch,
        }

        if not os.path.isdir('checkpoint/ResNet18/'):
            os.mkdir('checkpoint/ResNet18/')
        torch.save(state, './checkpoint/ResNet18/ckpt.pth_' + args.model + '_' + args.mixup + '_' + 'last_model.pth')
    test_rms = 100 * calib_err(rms_confidence, rms_correct, p='2')
    return (test_loss / batch_idx, 100. * correct / total, 100. * top1_acc / len(testloader), 100. * top5_acc / len(testloader), test_rms)


if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc', 'train top1 acc', 'train top5 acc', 'train_rms',
                            'test loss', 'test acc', 'test top1 acc', 'test top5 acc', 'test_rms'])

for epoch in range(start_epoch, args.epoch):
    train_loss, reg_loss, train_acc, train_top1_acc, train_top5_acc, train_rms = train(epoch)
    # test_loss, test_acc, wrong_img_list = test(epoch)
    test_loss, test_acc, test_top1_acc, test_top5_acc, test_rms = test(epoch)

    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, reg_loss, train_acc, train_top1_acc, train_top5_acc, train_rms, test_loss,
                            test_acc, test_top1_acc, test_top5_acc, test_rms])
        scheduler.step()
