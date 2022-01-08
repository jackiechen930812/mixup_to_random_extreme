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
    ResNet18_baseline  
    ResNet18_v2_epoch200
    ResNet18_v2_epoch1000
    DenseNet190
    DenseNet190_v2_epoch200

* ./mixup.py  : original mixup function
* ./mixup_v2.py : modified mixup function

* ./train.py  
    (Modify line 25 during training. If you use the original mixup, change it to import mixup as mp
            Use the modified mixup to import mixup_v2 as mp)
    

## Training
Use `python train.py` to train a new model.
Here is an example setting:
```
$ CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.1 --seed=20170922 --decay=1e-4
```

## 
