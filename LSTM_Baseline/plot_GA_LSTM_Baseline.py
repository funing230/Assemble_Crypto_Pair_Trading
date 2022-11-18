import numpy as np
import matplotlib.pyplot as plt

GA_ytest_pred = np.loadtxt(open("../result/GA_ytest_pred.csv", "rb"), delimiter=",", skiprows=0)
GA_ytest = np.loadtxt(open("../result/CNN_baseline_ytest.csv", "rb"), delimiter=",", skiprows=0)

LSTM1103_baseline_ytest_pred = np.loadtxt(open("../result/LSTM1103_baseline_ytest_pred.csv", "rb"), delimiter=",", skiprows=0)
baseline = np.loadtxt(open("../result/GA_ytest.csv", "rb"), delimiter=",", skiprows=0)
GA_LSTM_Prediction=np.loadtxt(open("../result/GA_ytest_pred.csv", "rb"), delimiter=",", skiprows=0)
CNN_baseline_ytest_pred=np.loadtxt(open("../result/CNN_baseline_ytest_pred.csv", "rb"), delimiter=",", skiprows=0)
GA_CNN_ytest_pred=np.loadtxt(open("../result/GA_CNN_ytest_pred.csv", "rb"), delimiter=",", skiprows=0)

CNNLSTM_ytest_pred=np.loadtxt(open("../result/CNNLSTM_ytest_pred.csv", "rb"), delimiter=",", skiprows=0)
CNNLSTM_ytest=np.loadtxt(open("../result/CNNLSTM_ytest.csv", "rb"), delimiter=",", skiprows=0)

#GA_CNN_ytest_pred.csv

# plot baseline and predictions
plt.figure(figsize=(15, 8))
plt.title('Testing')
plt.plot(baseline, label="True Lable",color='RED')
plt.plot(CNN_baseline_ytest_pred, label="CNN_Prediction",color='Black')
plt.plot(GA_CNN_ytest_pred, label="GA_CNN_Prediction",color='Blue')
plt.plot(GA_LSTM_Prediction, label="GA_LSTM_Prediction",color='Green')

plt.plot(CNNLSTM_ytest_pred, label="CNNLSTM_ytest_pred",color='Orange')
# plt.plot(CNNLSTM_ytest, label="CNNLSTM_ytest",color='Green')

plt.legend()
plt.show()





# # split the dataset by maintaining the ratio of training to testing as 70:30 with random state as 2003
# x_train, x_test, y_train, y_test = train_test_split(X_total, y_total, test_size=0.3, random_state=2003)
# # convert the values of training set to the numpy array
# x_train_np = x_train
# y_train_np = y_train
# # convert the values of testing set to the numpy array
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