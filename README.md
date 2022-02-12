## Requirements and Installation
* A computer running macOS or Linux
* For training new models, you'll also need a NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* Python version 3.6
* A [PyTorch installation](http://pytorch.org/)

## FILE description
* ./checkpoint  : Stored with trained models

* ./results   : Stored training log

* ./mixup.py  :   original mixup function: Beta Distribution with vector

* ./mixup_v2.py : modified mixup function: Matrix-Mixup + Gaussian Distribution

* ./train.py  : randomly choose mix-up method
    (Modify line 25 during training. If you use the original mixup, change it to import mixup as mp
            Use the modified mixup to import mixup_v2 as mp)
* ./train_concat_all  : concatenate matrix mix-up images, original mix-up images and original images in on iteration
* ./train_one_third.py  : one_third concatenation


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
