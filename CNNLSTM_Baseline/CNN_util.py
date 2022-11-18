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
# import SGD for optimizer
from torch.optim import SGD
# import Adam for optimizer
from torch.optim import Adam
# to measure the performance import L1Loss
from torch.nn import L1Loss
# install pytorch's ignite and then import R2 score package
from ignite.contrib.metrics.regression.r2_score import R2Score
import math
from numpy import array
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os
import random
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

# define the method for calculating average L1 Loss and R2 Score of given model
def model_loss(model, dataset, train=False, optimizer=None):
    # first calculated for the batches and at the end get the average
    performance = L1Loss()
    score_metric = R2Score()
    avg_loss = 0
    avg_score = 0
    count = 0
    for input, output in iter(dataset):
        # get predictions of the model for training set
        predictions = model.feed(input)

        # calculate loss of the model
        loss = performance(predictions, output)

        # compute the R2 score
        score_metric.update([predictions, output])
        score = score_metric.compute()

        if (train):
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



# defined model named as CnnRegressor and
# this model should be the subclass of torch.nn.Module
class CnnRegressor(torch.nn.Module):
    # defined the initialization method
    def __init__(self, batch_size, inputs, outputs):
        # initialization of the superclass
        super(CnnRegressor, self).__init__()
        # store the parameters
        self.batch_size = batch_size
        self.inputs = inputs
        self.outputs = outputs
        # define the input layer
        self.input_layer = Conv1d(inputs, batch_size, 1, stride=1)

        # define max pooling layer
        self.max_pooling_layer = MaxPool1d(1)

        # define other convolutional layers
        self.conv_layer1 = Conv1d(batch_size, 128, 1, stride=3)
        self.conv_layer2 = Conv1d(128, 256, 1, stride=3)
        self.conv_layer3 = Conv1d(256, 512, 1, stride=3)

        # define the flatten layer
        self.flatten_layer = Flatten()

        # define the linear layer
        self.linear_layer = Linear(512, 128)

        # define the output layer
        self.output_layer = Linear(128, outputs)

    # define the method to feed the inputs to the model
    def feed(self, input):
        # input is reshaped to the 1D array and fed into the input layer
        input = input.reshape((self.batch_size, self.inputs, 1))
        input=input.to(device)
        # ReLU is applied on the output of input layer
        output = relu(self.input_layer(input))

        # max pooling is applied and then Convolutions are done with ReLU
        output = self.max_pooling_layer(output)
        output = relu(self.conv_layer1(output))

        output = self.max_pooling_layer(output)
        output = relu(self.conv_layer2(output))

        output = self.max_pooling_layer(output)
        output = relu(self.conv_layer3(output))

        # flatten layer is applied
        output = self.flatten_layer(output)

        # linear layer and ReLu is applied
        output = relu(self.linear_layer(output))

        # finally, output layer is applied
        output = self.output_layer(output)
        return output



class CnnOptimization:
    def __init__(self,model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    def train_step(self, dataset):
        # first calculated for the batches and at the end get the average
        performance = self.loss_fn
        score_metric = R2Score()
        avg_loss = 0
        avg_score = 0
        count = 0
        for input, output in iter(dataset):
            # get predictions of the model for training set
            predictions = self.model.feed(input)
            # calculate loss of the model
            loss = performance(predictions.to(device), output.to(device))
            # compute the R2 score
            score_metric.update([predictions.to(device), output.to(device)])
            score = score_metric.compute()
            # clear the errors
            self.optimizer.zero_grad()
            # compute the gradients for optimizer
            loss.backward()
            # use optimizer in order to update parameters
            # of the model based on gradients
            self.optimizer.step()
            # store the loss and update values
            avg_loss += loss.item()
            avg_score += score
            count += 1
        return avg_loss / count, avg_score / count

    def train(self,train_loader, n_epochs=50):
        for epoch in range(n_epochs):
            # model is cycled through the batches
            avg_loss, avg_r2_score = self.train_step(train_loader)
            if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {avg_loss:.8f}"
                )

    def evaluate(self,test_loader):
        # first calculated for the batches and at the end get the average
        performance = L1Loss()
        score_metric = R2Score()
        avg_loss = 0
        avg_score = 0
        count = 0
        predictions = []
        values = []
        for input, output in iter(test_loader):
            # input = input.view([batch_size, n_features])
            # get predictions of the model for training set
            self.model.eval()
            prediction = self.model.feed(input)
            predictions.append(prediction.cpu().detach().numpy())
            values.append(output.cpu().detach().numpy())
        return predictions, values

    def format_predictions(self,predictions, values, scaler):
        vals = np.concatenate(values, axis=0).ravel()
        preds = np.concatenate(predictions, axis=0).ravel()

        y_test_pred = scaler.inverse_transform(preds.reshape(-1, 1))
        y_test = scaler.inverse_transform(vals.reshape(-1, 1))
        fitness = math.sqrt(mean_squared_error(y_test[:, 0], y_test_pred[:, 0]))

        return y_test_pred,y_test,fitness



class MyDataset(Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
    def __len__(self):
        return self.data_tensor.size(0)   #.shape
    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]




def get_data(X_ss, y_en,shuffle=False):

    # train_test_cutoff = round(0.30 * total_samples) * -1

    # X_train = X_ss[:train_test_cutoff]
    # X_test = X_ss[train_test_cutoff:]
    # y_train = y_en[:train_test_cutoff]
    # y_test = y_en[train_test_cutoff:]

    x_train, x_test, y_train, y_test = train_test_split(X_ss, y_en, test_size=0.3,shuffle=shuffle)
    # make training and test sets in torch
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    train = MyDataset(x_train, y_train)
    test = MyDataset(x_test, y_test)

    return train,test

# split a multivariate sequence into samples
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


class CNNLSTM(torch.nn.Module):
    def __init__(self, batch_size,input_size, output_size, hidden_size, num_layers,dropout=0):
        super(CNNLSTM, self).__init__()
        self.batch_size = batch_size
        self.inputs = input_size
        self.outputs = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.conv1 = torch.nn.Conv1d(self.inputs,self.batch_size,1, stride=1)
        self.max_pooling_layer = MaxPool1d(1)
        self.conv2 = torch.nn.Conv1d(self.batch_size, 32, 1, stride=3)
        self.batch1 = torch.nn.BatchNorm1d(32)
        self.conv3 = torch.nn.Conv1d(32, 32, 1, stride=3)
        self.batch2 = torch.nn.BatchNorm1d(32)

        self.LSTM = torch.nn.LSTM(input_size=self.inputs, hidden_size=self.hidden_size,num_layers=self.num_layers,dropout=dropout, batch_first=True)
        self.fc1 = torch.nn.Linear(32 * hidden_size, output_size)
        # self.fc2 = nn.Linear(1, 1)

    def forward(self, x):
        # in_size1 = x.size(0)  # one batch
        x = x.reshape((self.batch_size, self.inputs, self.hidden_size))
        x = F.selu(self.conv1(x))
        # x = self.max_pooling_layer(x)
        x = self.conv2(x)
        x = F.selu(self.batch1(x))
        x = self.conv3(x)
        x = F.selu(self.batch2(x))
#----------------------add by funing
        # # Initialize hidden state with zeros
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # # Initialize cell state
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # # Index hidden state of last time step
        # output = self.fc(out[:, -1, :])
# ----------------------add by funing
        x, h = self.LSTM(x)
        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
        # in_size1 = x.size(0)  # one batch
        # x = x.view(in_size1, -1)
        # flatten the tensor x[:, -1, :]
        x = self.fc1(x)
        output = torch.sigmoid(x)
        # output = self.fc2(x)

        return output
