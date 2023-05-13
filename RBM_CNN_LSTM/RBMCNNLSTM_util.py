import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.metrics import mean_absolute_error
import torch
import random
from numpy import array
from torch.utils.data import Dataset, DataLoader
from RBM import *
import warnings

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

def get_Data(data_path):
    # totaldataset_file_path = 'total_dataset.csv'
    totaldataset_df = pd.read_csv(data_path, parse_dates=[0], index_col=0)

    X, y = totaldataset_df.drop(columns=['close']), totaldataset_df['close']
    return X, y

def new_get_Data(data_path):
    # totaldataset_file_path = 'total_dataset.csv'
    totaldataset_df = pd.read_csv(data_path, parse_dates=[0], index_col=0)

    X, y = totaldataset_df.drop(columns=['close']), totaldataset_df['close']

    y=y.shift(1).dropna();
    X=X.drop(X.index[-1])

    return X, y

def get_Data():
    totaldataset_df_BTC = 'totaldataset_df_BTC.csv'
    BTC_df = pd.read_csv(totaldataset_df_BTC, parse_dates=[0], index_col=0)
    totaldataset_df_ETH = 'totaldataset_df_ETH.csv'
    ETH_df = pd.read_csv(totaldataset_df_ETH, parse_dates=[0], index_col=0)
    all_df = pd.concat([BTC_df, ETH_df], axis=1)
    all_df = all_df.loc['2018-10-04':]
    all_df = all_df.drop(all_df.index[-1])
    z_score_singel = 'z_score_singel.csv'
    singel_df = pd.read_csv(z_score_singel)

    X = all_df
    y = singel_df['z_score_singel']

    return X, y



# 数据预处理
def normalization(data,label):

    mm_x=MinMaxScaler(feature_range=(-1, 1)) # 导入sklearn的预处理容器
    mm_y=MinMaxScaler(feature_range=(-1, 1))
    # data=data.values    # 将pd的系列格式转换为np的数组格式
    label=np.array(label).reshape(-1, 1)
    data=mm_x.fit_transform(data) # 对数据和标签进行归一化等处理
    label=mm_y.fit_transform(label.reshape(-1, 1))
    return data,label,mm_y


def normalization_x(data):

    mm_x=MinMaxScaler(feature_range=(-1, 1)) # 导入sklearn的预处理容器

    # data=data.values    # 将pd的系列格式转换为np的数组格式

    data=mm_x.fit_transform(data) # 对数据和标签进行归一化等处理

    return data

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

class MyDataset(Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
    def __len__(self):
        return self.data_tensor.size(0)   #.shape
    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]


def get_split_data(X_ss, y_en,shuffle=False):

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




class Net(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, num_layers, output_size, batch_size, kernel_size,seq_length,dropout=0.2) -> None:
        super(Net, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.dropout=dropout
        self.seq_length=seq_length
        self.num_directions = 1  # 单向LSTM
        self.relu = nn.ReLU(inplace=True)
        # (batch_size=64, seq_len=3, input_size=3) ---> permute(0, 2, 1)
        # (64, 3, 3)

        #num_visible, num_hidden, k, learning_rate=1e-3, momentum_coefficient=0.5, weight_decay=1e-4, use_cuda=True
        self.rbm=RBM(visible_units=(self.in_channels),
                              hidden_units=self.out_channels*100,
                              k=2,
                              learning_rate=0.02,
                              learning_rate_decay=False,
                              increase_to_cd_k=False, device=device)

        self.conv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels*100, kernel_size=self.kernel_size,device=device )
        self.lstm = nn.LSTM(input_size=(self.seq_length-self.kernel_size+1+self.seq_length),
                            hidden_size=self.hidden_size, dropout=self.dropout,
                            num_layers=self.num_layers, batch_first=True).to(device)
        self.fc = nn.Linear(self.hidden_size, self.output_size).to(device)

    def forward(self, x):
        rbm_x,_ = self.rbm(x.to(device))
        rbm_x = rbm_x.permute(0, 2, 1)
        x = x.permute(0, 2, 1)
        con_x = self.conv(x.to(device))
        # con_x = con_x.permute(0, 2, 1)
        # batch_size, seq_len = x.size()[0], x.size()[1]
        x_all = torch.cat([rbm_x, con_x.to(device)], 2)
        h_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size).to(device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(x_all.to(device), (h_0, c_0))
        pred = self.fc(output)
        pred = pred[:, -1, :]
        return pred

class CNNLSTMOptimization:
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
        loss = loss_func(y.to(device), yhat.to(device))
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
                predictions.append(yhat.detach().cpu().numpy())
                values.append(y_test.detach().cpu().numpy())
        return predictions, values

    def format_predictions(self,predictions, values, scaler):
        vals = np.concatenate(values, axis=0).ravel()
        preds = np.concatenate(predictions, axis=0).ravel()

        y_test_pred = scaler.inverse_transform(preds.reshape(-1, 1))
        y_test = scaler.inverse_transform(vals.reshape(-1, 1))
        fitness = math.sqrt(mean_squared_error(y_test[:, 0], y_test_pred[:, 0]))

        return y_test_pred,y_test,fitness

class Net_classification(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, num_layers, output_size, batch_size, kernel_size,
                 seq_length, dropout=0.2) -> None:
        super(Net_classification, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.seq_length = seq_length
        self.num_directions = 1  # 单向LSTM
        self.relu = nn.ReLU(inplace=True)

        self.rbm = RBM(visible_units=(self.in_channels),
                       hidden_units=self.out_channels * 100,
                       k=2,
                       learning_rate=0.02,
                       learning_rate_decay=False,
                       increase_to_cd_k=False, device=device)

        self.conv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels * 100,
                              kernel_size=self.kernel_size, device=device)
        self.lstm = nn.LSTM(input_size=(self.seq_length - self.kernel_size + 1 + self.seq_length),
                            hidden_size=self.hidden_size, dropout=self.dropout,
                            num_layers=self.num_layers, batch_first=True).to(device)
        self.fc = nn.Linear(self.hidden_size, self.output_size).to(device)
        self.softmax = nn.Softmax(dim=1).to(device)

    def forward(self, x):
        rbm_x, _ = self.rbm(x.to(device))
        rbm_x = rbm_x.permute(0, 2, 1)
        x = x.permute(0, 2, 1)
        con_x = self.conv(x.to(device))
        x_all = torch.cat([rbm_x, con_x.to(device)], 2)
        h_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size).to(device)
        output, _ = self.lstm(x_all.to(device), (h_0, c_0))
        pred = self.fc(output)
        pred = self.softmax(pred)
        return pred

class Net_classification_optimization:
    def __init__(self, model, loss_fn, optimizer):
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
        loss = self.loss_fn(yhat.to(device), y.to(device))

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(self, train_loader, n_epochs=50):
        for epoch in range(1, n_epochs + 1):
            batch_losses = []

            for x_batch, y_batch in train_loader:
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)

            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            if (epoch <= 10) or (epoch % 50 == 0):
                print(f"[{epoch}/{n_epochs}] Training loss: {training_loss:.8f}")

    def evaluate_for_one(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []

            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features])
                y_test = y_test.long()

                self.model.eval()
                yhat = self.model(x_test)

                # Get predicted classes by taking the argmax of the output
                # of the model
                _, predicted = torch.max(yhat.data, 1)

                predictions.append(predicted.detach().cpu().numpy())
                values.append(y_test.detach().cpu().numpy())

        return predictions, values

    def format_predictions(self, predictions, values):
        # Concatenate values and predictions
        vals = np.concatenate(values).ravel()
        preds = np.concatenate(predictions).ravel()

        # Calculate accuracy
        accuracy = accuracy_score(vals, preds)

        return preds, vals, accuracy
