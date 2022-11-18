# import pandas library
import pandas as pd
# import pyplot for plotting graph
import matplotlib.pyplot as plt
# import library numpy which helps to manipulate data
import numpy as np
# import package to split the given data into training and testing set
from sklearn.model_selection import train_test_split
import torch
# import 1D convolutional layer
from torch.nn import Conv1d
# import max pooling layer
from torch.nn import MaxPool1d
# import the flatten layer
from torch.nn import Flatten
# import linear layer
from torch.nn import Linear
# import activation function (ReLU)
from torch.nn.functional import relu
# import libraries required for working with dataset from pytorch
from torch.utils.data import DataLoader, TensorDataset
# import SGD for optimizer
from torch.optim import SGD
from sklearn.preprocessing import MinMaxScaler
# import Adam for optimizer
from torch.optim import Adam
# to measure the performance import L1Loss
from torch.nn import L1Loss
from ignite.contrib.metrics.regression.r2_score import R2Score
from CNN_util import CnnRegressor,model_loss,CnnOptimization,seed_everything,get_data
import warnings

warnings.filterwarnings("ignore")
seed = 42
seed_everything(seed)


totaldataset_file_path = '../Dataset/total_dataset.csv'
totaldataset_df = pd.read_csv(totaldataset_file_path, parse_dates=[0], index_col=0)
X, y = totaldataset_df.drop(columns=['close']), totaldataset_df['close']

# Normalization------------------------------Data Processing----------------------
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler_y = MinMaxScaler(feature_range=(-1, 1))
# X = X.values
y = np.array(y).reshape(-1, 1)
X_total = scaler.fit_transform(X)  # ss.fit_transform(X)  normalize(X)
y_total = scaler_y.fit_transform(y)
# scaler = Normalizer()
# scaler_y = Normalizer()
# X_trans=scaler.fit_transform(X,window=window)
# y_trans=scaler_y.fit_transform(y,window=window)
# END---------------Normalization------------------------------Data Processing----------------------
total_samples = len(X)
train, test=get_data(X_total, y_total,shuffle=False)
batch_size = 4


# # split the dataset by maintaining the ratio of training to testing as 70:30 with random state as 2003
# x_train, x_test, y_train, y_test = train_test_split(X_total, y_total, test_size=0.3, random_state=2003)
# # convert the values of training set to the numpy array
# x_train_np = x_train
# y_train_np = y_train
# # convert the values of testing set to the  numpy array
# x_test_np = x_test
# y_test_np = y_test
# # to process with GPU, training set is converted into torch variable
# train_inputs = torch.from_numpy(x_train_np).cuda().float()
# train_outputs = torch.from_numpy(y_train_np.reshape(y_train_np.shape[0], 1)).cuda().float()
# # create the DataLoader instance to work with batches
# train_tensor = TensorDataset(train_inputs, train_outputs)
# train_loader = DataLoader(train_tensor, batch_size, shuffle=True, drop_last=True,num_workers=0)
# # to process with GPU, testing set is converted into torch variable
# test_inputs = torch.from_numpy(x_test_np).cuda().float()
# test_outputs = torch.from_numpy(y_test_np.reshape(y_test_np.shape[0], 1)).cuda().float()
# # create the DataLoader instance to work with batches
# test_tensor = TensorDataset(test_inputs, test_outputs)
# test_loader = DataLoader(test_tensor, batch_size, shuffle=False, drop_last=True,num_workers=0)
# # test_loader_one = DataLoader(dataset=test_tensor, batch_size=1, shuffle=False, num_workers=0)
train_loader = DataLoader(dataset=train,batch_size=batch_size,shuffle=True,num_workers=0, drop_last=True)
test_loader = DataLoader(dataset=test,batch_size=batch_size,shuffle=False,num_workers=0, drop_last=True)

imput_size=X.shape[1]
output_size=1
# define the batch size
model = CnnRegressor(batch_size, imput_size, output_size)
# we are using GPU so we have to set the model for that
model.cuda()
# define the number of epochs
network_epoch = 200
loss_fn =torch.nn.L1Loss()
optimizer = Adam(model.parameters(), lr=0.01)

opt = CnnOptimization(model=model,loss_fn=loss_fn, optimizer=optimizer)
opt.train(train_loader, n_epochs=network_epoch)
predictions, values = opt.evaluate(test_loader)

y_test_pred,y_test,fitness = opt.format_predictions(predictions, values,scaler_y)
# temp=eval(y_test_pred)
np.savetxt("CNN_baseline_ytest_pred.csv", y_test_pred, delimiter=',',fmt='%f')
np.savetxt("CNN_baseline_ytest.csv",y_test, delimiter=',',fmt='%f')

print('fitness``````',fitness)

# torch.save(model, "./model/" + str(fitness)+ "_weight.pth")


# print(predictions)
# print(values)
# # output of the performance of the model
# avg_loss, avg_r2_score = model_loss(opt.model, test_loader)
# print("The model's L1 loss is: " + str(avg_loss))
# print("The model's R^2 score is: " + str(avg_r2_score))


