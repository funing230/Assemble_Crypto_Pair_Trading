# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 11:17:43 2020

@author: Kianoosh Keshavarzian
"""

import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn


def split_sequences(input_sequences, output, n_steps_in):
    X, y = list(), list()
    for i in range(len(input_sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        # check if we are beyond the sequence
        if end_ix > len(input_sequences) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = input_sequences[i:end_ix], output[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# reading data frame ==================================================
totaldataset_file_path = '../Dataset/total_dataset.csv'  # '20181001':'20211201'  “time,high,low,open,volumefrom,volumeto,close”
totaldataset_df = pd.read_csv(totaldataset_file_path)  # , parse_dates=[0], index_col=0

X, y = totaldataset_df.drop(columns=['time', 'close']), totaldataset_df['close']

# Normalization------------------------------Data Processing----------------------
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler_y = MinMaxScaler(feature_range=(-1, 1))
X = X.values
y = np.array(y).reshape(-1, 1)
X_total = scaler.fit_transform(X)  # ss.fit_transform(X)  normalize(X)
y_total = scaler_y.fit_transform(y)

# print(X_trans)
# Cut dataset by time step
time_step = 1  # [1,2,3,4,5,6.......30]=====>[31]
X_ss, y_en = split_sequences(X_total, y_total, time_step)

# total_samples = len(X)
# train_test_cutoff = round(0.30 * total_samples) * -1
# val_size = 100
# X_train = X_ss[:train_test_cutoff]
# X_test = X_ss[train_test_cutoff:]
# y_train = y_en[:train_test_cutoff]
# y_test = y_en[train_test_cutoff:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_ss, y_en, test_size=0.3, shuffle=False)


# make training and test sets in torch
x_train = torch.from_numpy(X_train).type(torch.Tensor)
x_test = torch.from_numpy(X_test).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor)
y_test = torch.from_numpy(y_test).type(torch.Tensor)

# Build model
##################################################

input_dim = 32
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 200


# Here we define our model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # Building your LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        out = self.fc(out[:, -1, :])

        return out


model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

loss_fn = torch.nn.MSELoss(size_average=True)  # size_average=True

optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
print(model)
print(len(list(model.parameters())))
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())

# Train model
##################################################################

hist = np.zeros(num_epochs)

for t in range(num_epochs):
    # Initialise hidden state

    # Forward pass
    y_train_pred = model(x_train)

    loss = loss_fn(y_train_pred, y_train)
    if t % 10 == 0 and t != 0:
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    # Zero out gradient, else they will accumulate between epochs
    optimiser.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimiser.step()


import sys
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger(stream=sys.stdout)
# now it works
print('print something')
print("output")



plt.plot(y_train_pred.detach().numpy(), label="Preds")
plt.plot(y_train.detach().numpy(), label="Data")
plt.legend()
plt.show()

plt.plot(hist, label="Training loss")
plt.legend()
plt.show()

# make predictions
y_test_pred = model(x_test)

# invert predictions
y_train_pred = scaler_y.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler_y.inverse_transform(y_train.detach().numpy())
y_test_pred = scaler_y.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler_y.inverse_transform(y_test.detach().numpy())

np.savetxt("LSTM1103_baseline_ytest_pred.csv", y_test_pred, delimiter=',',fmt='%f')
np.savetxt("lstm_baseline_ytest.csv", y_test, delimiter=',')

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train[:, 0], y_train_pred[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[:, 0], y_test_pred[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))

# plot baseline and predictions
plt.figure(figsize=(15, 8))
plt.title('Training')
# plt.plot(y_train_pred.detach().numpy(), label="Preds")
plt.plot(y_train_pred, label="Preds")
plt.plot(y_train, label="Data")
plt.legend()
plt.show()

# plot baseline and predictions
plt.figure(figsize=(15, 8))
plt.title('Testing')
plt.plot(y_test_pred, label="Preds")
plt.plot(y_test, label="Data")
plt.legend()
plt.show()
