# univariate mlp example
# from timeseries_util import split_sequences,LSTM,fitness_calculate,MyDataset
from CNNLSTM_Baseline.GA_util import LSTM,Optimization, split_sequences,seed_everything,get_data
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import warnings


warnings.filterwarnings("ignore")
seed = 42
seed_everything(seed)

n_steps = 2

totaldataset_file_path = '../Dataset/total_dataset.csv'
totaldataset_df = pd.read_csv(totaldataset_file_path, parse_dates=[0], index_col=0)

X, y = totaldataset_df.drop(columns=['close']), totaldataset_df['close']

# Normalization------------------------------Data Processing----------------------
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler_y = MinMaxScaler(feature_range=(-1, 1))
# X = X.values
y = np.array(y).reshape(-1, 1)
X_trans = scaler.fit_transform(X)  # ss.fit_transform(X)  normalize(X)
y_trans = scaler_y.fit_transform(y)

# scaler = Normalizer()
# scaler_y = Normalizer()
# X_trans=scaler.fit_transform(X,window=window)
# y_trans=scaler_y.fit_transform(y,window=window)



# END---------------Normalization------------------------------Data Processing----------------------

# Cut dataset by time step[1,2,3,4,5,6.......30]=====>[31]
X_ss, y_en = split_sequences(X_trans, y_trans, n_steps)
# print(X_ss)
# print("Create time step :",totaldataset_df.shape)
total_samples = len(X)
train, test=get_data(X_ss, y_en,False)

#Model parameter
input_dim = 32   #total feature number
output_dim = 1

batch_size=4
hidden_dim = 32
num_layers = 4
dropout = 0.4

train_dataloader = DataLoader(dataset=train,batch_size=batch_size,shuffle=True,num_workers=0)
test_dataloader = DataLoader(dataset=test,batch_size=batch_size,shuffle=True,num_workers=0)
test_dataloader_one = DataLoader(dataset=test, batch_size=1, shuffle=False, num_workers=0)

# Model construction
model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers,dropout=dropout)
loss_fn =torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
# Compile model
# -----------------------------------------------------------------for Traing
network_epoch = 200
opt = Optimization(model=model,loss_fn=loss_fn, optimizer=optimiser)
opt.train(train_dataloader, n_epochs=network_epoch)
predictions, values = opt.evaluate_for_one(test_dataloader_one,1,input_dim)
y_test_pred,y_test,fitness = opt.format_predictions(predictions, values,scaler_y)

np.savetxt("./result/LSTM1107_baseline_ytest_pred.csv", y_test_pred, delimiter=',',fmt='%f')
# save model weights funing
# torch.save(model, "./model/" + str(fitness)+ "_weight.pth")

print("---batch_size---:",batch_size,"---n_steps---:",n_steps,"---network_epoch---:",network_epoch)
print("---hidden_dim---:",hidden_dim,"---dropout---:",dropout,"---num_layers---:",num_layers)
print("----FITNESS-----------RMSE----",fitness,"----------")




