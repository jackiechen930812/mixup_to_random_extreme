import torch
import torchvision.transforms as transforms
import torchvision.utils
from torch.utils.data import DataLoader
from tqdm import tqdm

transform_test = transforms.Compose([
    transforms.ToTensor(),  # 255 1
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

saved_model_path = 'checkpoint/ResNet18/cifar100/ckpt.pth_ResNet18_none_last_model.pth'
# saved_model_path = 'checkpoint/ResNet18/cifar100/ckpt.pth_ResNet18_cutmix_last_model.pth'
# saved_model_path = 'checkpoint/ResNet18/cifar100/ckpt.pth_ResNet18_ori_last_model.pth'
# saved_model_path = 'checkpoint/ResNet18/cifar100/ckpt.pth_ResNet18_comix_last_model.pth'
# saved_model_path = 'checkpoint/ResNet18/cifar100/ckpt.pth_ResNet18_puzzlemix_last_model.pth'
# saved_model_path = 'checkpoint/ResNet18/cifar100/ckpt.pth_ResNet18_one_fourth_last_model.pth'

trainset = torchvision.datasets.ImageFolder('./transfer_res/images', transform=transform_test)

testloader = torch.utils.data.DataLoader(trainset, batch_size=10,
                                         shuffle=False, num_workers=8)

checkpoint = torch.load(saved_model_path)
net = checkpoint['net']
net.eval().cuda()

# 模型在原测试集上的准确率
correct = 0
total = 0

for images, labels in tqdm(testloader):
    images = images.cuda()
    outputs = net(images)

    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print(', test accuracy: %.2f %%' % (100 * float(correct) / total))
