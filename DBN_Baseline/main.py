import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from DBN import DBN
from sklearn.preprocessing import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# Set parameter

device = 'cuda'

if device == 'cuda':
    assert torch.cuda.is_available() is True, "cuda isn't available"
    print('Using GPU backend.\n'
          'GPU type: {}'.format(torch.cuda.get_device_name(0)))
else:
    print('Using CPU backend.')

# train & predict
# data
input_length = 32
output_length = 1
test_percentage = 0.3
# network
hidden_units = [256]
batch_size = 4
epoch_pretrain = 100
epoch_finetune = 50
# Gibbs_sampling_step=5
# learning_rate=0.0001
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam

# Generate input and output data
totaldataset_file_path = 'total_dataset.csv'
totaldataset_df = pd.read_csv(totaldataset_file_path, parse_dates=[0], index_col=0)

X, y = totaldataset_df.drop(columns=['close']), totaldataset_df['close']
# Normalization------------------------------Data Processing----------------------
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler_y = MinMaxScaler(feature_range=(-1, 1))
# X = X.values
y = np.array(y).reshape(-1, 1)
X_total = scaler.fit_transform(X)  # ss.fit_transform(X)  normalize(X)
y_total = scaler_y.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(X_total, y_total, test_size=0.3,shuffle=False)

print('x_train.shape:' + str(x_train.shape))
print('y_train.shape:' + str(y_train.shape))
print('x_test.shape:' + str(x_test.shape))
print('y_test.shape' + str(y_test.shape))

# Build model   hidden_units, visible_units=NO, output_units=1, k=1-10,learning_rate=0.001,
dbn = DBN(hidden_units, input_length, output_length, device=device)

# Train model
dbn.pretrain(x_train, epoch=epoch_pretrain, batch_size=batch_size)
dbn.finetune(x_train, y_train, epoch_finetune, batch_size, loss_function,optimizer(dbn.parameters()))
# torch.save(dbn, 'dbn.pth')
# Make prediction and plot
y_predict = dbn.predict(x_test, batch_size)
y_real = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_predict = scaler_y.inverse_transform(y_predict.reshape(-1, 1)).flatten()
plt.figure(figsize=(15, 8))
plt.plot(y_predict, label='prediction')
plt.plot(y_real, label='real')
plt.xlabel('MSE Error: {}'.format(mean_squared_error(y_real, y_predict)))
plt.legend()
plt.title('Prediction result')
plt.savefig('dbn_prediction.png')
plt.show()

