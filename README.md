## 生成低频高频攻击数据：frequency.py
* 代码基本都是来自github上frequencyHelper.py ，对数据保存的地址进行了修改
* 代码108行加了一行生成测试数据标签的代码，方便后续模型测试
* 运行：
    python frequency.py
    生成数据存放与./data/CIFAR10/下

## 创建新的dataset的类： new_dataset.py
* 用于读取frequency.py生成的数据，并处理为能被dataloader加载的形式
* 由于github代码上会将几个生成的数据拼接为一个(如train_data_low_4和train_data_high_4等相拼接)，所以在new_dataset.py中加了一个__add__函数
* 代码内容见注释

## 在train.py中使用生成的数据进行训练
* 65-103行为新添加的部分
* 65行use_frequency_data参数控制是否使用生成的数据，True使用，False使用原数据
* 将train_data_low_4.npy train_data_high_4.npy 及73行radius列表中对应后缀的npy数据进行拼接,之后放入dataloader
* test时加载与train相同的数据后缀
* 其他：
    28行中引入创建的New_Dataset类
    178-179行、197-201行将target转化为long的格式，cutmix在cutmix.py的52行已经进行了转化。 ————不加会报错

<!-- 
## Requirements and Installation
* A computer running macOS or Linux
* For training new models,  you'll also need a NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* Python version 3.6
* A [PyTorch installation](http://pytorch.org/)

## FILE description
* ./checkpoint  : Stored with trained models

* ./results   : Stored training log

* ./mixup.py  :   original mixup function: Beta Distribution with vector

* ./mixup_v2.py : modified mixup function: Matrix-Mixup + Gaussian Distribution

* ./train.py  : one_third concatenation(matrix mix-up images, original mix-up images and original images in one iteration

* ./baseline.py: original mixup with/without tricks (--use_cutmix=True) to enable cutmix

    


## Training
Use `python train.py` to train a new model.
Here is an example setting:
```
$ CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.1 --seed=20220103 --decay=1e-4
```

## Generate mix-up images
Uncomment Line :63,64,66,67 in train.py & Uncomment Line 30-33 in mixup_v2.py
```
$ python train.py --lr=0.1 --seed=20220103 --decay=1e-4 --epoch=1
```
## Add adv samples for test
1、install torchattacks
    pip install torchattacks

2、PGD_eval.py
   run with
```
    CUDA_VISIBLE_DEVICES=0 python PGD_eval.py
```

## 
