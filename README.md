## Requirements and Installation
* A computer running macOS or Linux
* For training new models, you'll also need a NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* Python version 3.6
* A [PyTorch installation](http://pytorch.org/)

## FILE description
* ./checkpoint  : Stored with trained models
    ResNet18 : Stored baseline best model of ResNet18 
    ResNet18_v2 : Stored the best model of ResNet18 using the modified mixup, including the results obtained by training 200 epochs by default and training 1000 epochs respectively
    DenseNet190 : Stored baseline best model of DenseNet190 
    DenseNet190_v2 : Stored the best model of DenseNet190 using the modified mixup

* ./results   : Stored training log
*      ./results/baseline
                 ResNet18_baseline  
                 ResNet18_v2_epoch200
*      ./results/sum_up

* ./mixup.py  :   original mixup function: Beta Distribution with vector
* ./mixup_v2.py : modified mixup function: Matrix-Mixup + Gaussian Distribution
* ./train.py  
    (Modify line 25 during training. If you use the original mixup, change it to import mixup as mp
            Use the modified mixup to import mixup_v2 as mp)

## Training
Use `python train.py` to train a new model.
Here is an example setting:
```
$ CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.1 --seed=20170922 --decay=1e-4
```

## Add Gaussian Noise for comparison 
* ./checkpoint/ResNet18/

    model1-1 original mix-up + Gaussian Noise
    
    model1-2 Matrix-Mixup + Gaussian Distribution + Gaussian Noise

    model2-1 original mix-up                      (ckpt.t7_ResNet18_epoch50_2_1_baseline_20220103)
    
    model2-2 Matrix-Mixup + Gaussian Distribution(mixup_v2)    (ckpt.t7_ResNet18_epoch50_2_2_gua_matrix_20220103)

## Add adv samples for test
1、install torchattacks
    pip install torchattacks

2、PGD_eval.py
   run with
```
    CUDA_VISIBLE_DEVICES=0 python PGD_eval.py
```

3、Test results with pgd attack: 
model2-1 test result：
Before PGD attack, accuracy: 91.58 %
After PGD attack, accuracy: 18.56 %

model2-2 test result：

    Before PGD attack, accuracy: 78.17 %

    After PGD attack, accuracy: 24.47 %

## 
