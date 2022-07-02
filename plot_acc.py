import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_csv(filename):
    data = pd.read_csv(filename)
    data_acc = data[['test acc']]
    data_epoch = data[['epoch']]
    x = np.array(data_epoch)
    y = np.array(data_acc)
    return x, y



x1, y1 = read_csv("./results/log_resnest50_epoch200_cutmix20220501.csv")
x2, y2 = read_csv("./results/log_resnest50_epoch200_matrix20220502.csv")
x3, y3 = read_csv("./results/log_resnest50_epoch200_none20220501.csv")
x4, y4 = read_csv("./results/log_resnest50_epoch200_ori20220502.csv")

plt.plot(x1,y1,label="test_acc_cutmix")
plt.plot(x1, y2, label="test_acc_matrix")
plt.plot(x1, y3, label="test_acc_none")
plt.plot(x1, y4, label="test_acc_ori")

plt.title("test ccuracy")
plt.xlabel('epoch')
plt.ylabel('test_acc')
plt.legend()
plt.show()