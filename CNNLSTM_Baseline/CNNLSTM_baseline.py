import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data as Data
from CNNLSTM_util import *
from torchvision import transforms, datasets
import warnings

warnings.filterwarnings("ignore")
seed = 42
seed_everything(seed)



# seq_length=6 # 时间步长
# input_size=32
# num_layers=6
# out_channels=8
# hidden_size=64
# batch_size=4
# n_iters=5000
# lr=0.001
# output_size=1
# split_ratio=0.7
# kernel_size=2
# network_epoch=100
# dropput=0.3

seq_length=7 # 时间步长
input_size=32
num_layers=4
out_channels=10
hidden_size=32
batch_size=30
n_iters=5000
lr=0.001
output_size=1
split_ratio=0.7
kernel_size=2
network_epoch=100
dropput=0.3


#-------------------add by funing for getdataset---------------------------------------
totaldataset_file_path = '../Dataset/total_dataset.csv'
X, y = get_Data(totaldataset_file_path)
X_total,y_total,scaler_y=normalization(X, y)

X_ss, y_en = split_sequences(X_total, y_total, seq_length)
train,test= get_split_data(X_ss, y_en, False)

train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, num_workers=0,drop_last=True)
test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True, num_workers=0,drop_last=True)
test_dataloader_one = DataLoader(dataset=test, batch_size=1, shuffle=False, num_workers=0,drop_last=True)
#-------------------add by funing for getdataset---------------------------------------------


#in_channels, out_channels, hidden_size, num_layers, output_size, batch_size, seq_length)
model=Net(input_size,out_channels,hidden_size,num_layers,output_size,batch_size,kernel_size,seq_length,dropput)
criterion=torch.nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=lr)
print(model)

opt = CNNLSTMOptimization(model=model, loss_fn=criterion, optimizer=optimizer)
opt.train(train_loader, n_epochs=network_epoch)
predictions, values = opt.evaluate_for_one(test_dataloader_one, 1, input_size)
y_test_pred,y_test,fitness = opt.format_predictions(predictions, values, scaler_y)

np.savetxt("Assemble_Model_pred.csv", y_test_pred, delimiter=',',fmt='%f')
np.savetxt("Baseline_ytest.csv",y_test, delimiter=',',fmt='%f')


# plot baseline and predictions
plt.figure(figsize=(15, 8))
plt.title('Testing')
plt.plot(y_test, label="True Lable",color='RED')
plt.plot(y_test_pred, label="Assemble_Model_pred",color='Black')
plt.xlabel('MSE Error: {}'.format(mean_squared_error(y_test, y_test_pred)))
plt.legend()
plt.title('Prediction result')
plt.savefig('dbn_prediction3.png')
plt.show()



