import numpy as np
from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import Dense
# from keras import optimizers
import statsmodels.datasets.co2 as co2
from permetrics.regression import RegressionMetric
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os
import random
from sklearn.model_selection import train_test_split
# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')


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


# Here we define our model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim,output_dim,num_layers,dropout ):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.num_layers = num_layers
        # Building your LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout,batch_first=True)
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # Initialize cell state
        temp=x.size(0)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # Index hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out


class Optimization:
    def __init__(self,model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    def train_step(self, x, y):
        # Sets model to train mode
        self.model.train()
        # Makes predictions
        yhat = self.model(x)
        # Computes loss
        loss_func = self.loss_fn
        loss = loss_func(y, yhat)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()
        # Returns the loss
        return loss.item()

    def train(self,train_loader, n_epochs=50):
        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                # x_batch = x_batch.view([batch_size, input_dim, n_features]).to(device)
                # x_batch = x_batch.to(device)
                # y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.8f}"
                )

    # def evaluate(self,test_loader):
    #     with torch.no_grad():
    #         predictions = []
    #         values = []
    #         for x_test, y_test in test_loader:
    #             # x_test = x_test.view([batch_size, -1, n_features])
    #             # y_test = y_test
    #             self.model.eval()
    #             yhat = self.model(x_test)
    #             predictions.append(yhat.detach().numpy())
    #             values.append(y_test.detach().numpy())
    #     return predictions, values

    def evaluate_for_one(self,test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features])
                y_test = y_test
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.detach().numpy())
                values.append(y_test.detach().numpy())
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


def get_data(X_ss, y_en,shuffle=False):

    # train_test_cutoff = round(0.30 * total_samples) * -1
    #
    # X_train = X_ss[:train_test_cutoff]
    # X_test = X_ss[train_test_cutoff:]
    # y_train = y_en[:train_test_cutoff]
    # y_test = y_en[train_test_cutoff:]
    x_train, x_test, y_train, y_test = train_test_split(X_ss, y_en, test_size=0.3,shuffle=False)

    # make training and test sets in torch
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    train = MyDataset(x_train, y_train)
    test = MyDataset(x_test, y_test)

    return train,test

class Normalizer():
    def __init__(self):
        self.mu = None
        self.sd = None
    def fit_transform(self, x,window):
        self.mu = x.rolling(window=window, min_periods=1).mean()
        self.sd = x.rolling(window=window, min_periods=1).std()
        normalized_x = (x - self.mu)/self.sd
        return normalized_x.dropna()
    def inverse_transform(self, x):
        return ((x*self.sd) + self.mu).dropna()