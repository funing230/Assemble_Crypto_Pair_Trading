# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 16:31:10 2022

@author: ozancan ozdemir
"""

import numpy as np
from numpy import array
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from CNNN_Baseline.CNN_util import get_data,CNNLSTM
from torch.utils.data import DataLoader
from ignite.contrib.metrics.regression.r2_score import R2Score
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


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
train_loader = DataLoader(dataset=train,batch_size=batch_size,shuffle=True,num_workers=0, drop_last=True)
test_loader = DataLoader(dataset=test,batch_size=batch_size,shuffle=False,num_workers=0, drop_last=True)
# Build model
##################################################

input_size = 32
hidden_size = 32
num_layers = 2
output_size = 1
num_epochs = 100




model = CNNLSTM(batch_size,input_size, output_size, hidden_size, num_layers)
print(model)

num_epochs = 8
learning_rate = 0.01
criterion = torch.nn.MSELoss()  # mean-squared error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)
loss_list = []
# Train the model

def train_step(dataset):
    # first calculated for the batches and at the end get the average
    performance = criterion
    score_metric = R2Score()
    avg_loss = 0
    avg_score = 0
    count = 0
    for input, output in iter(dataset):
        # get predictions of the model for training set
        predictions = model.forward(input)
        # calculate loss of the model
        loss = performance(predictions.to(device), output.to(device))
        # compute the R2 score
        score_metric.update([predictions.to(device), output.to(device)])
        score = score_metric.compute()
        # clear the errors
        optimizer.zero_grad()
        # compute the gradients for optimizer
        loss.backward()
        # use optimizer in order to update parameters
        # of the model based on gradients
        optimizer.step()
        # store the loss and update values
        avg_loss += loss.item()
        avg_score += score
        count += 1
    return avg_loss / count, avg_score / count

for epoch in range(num_epochs):
    avg_loss, avg_r2_score = train_step(train_loader)
    if (epoch <= 10) | (epoch % 50 == 0):
        print(
            f"[{epoch}/{num_epochs}] Training loss: {avg_loss:.8f}"
        )