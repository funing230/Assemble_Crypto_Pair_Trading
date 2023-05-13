import torch.optim

from RBMCNNLSTM_util import *
from torchvision import transforms, datasets
import warnings

warnings.filterwarnings("ignore")
seed = 42
seed_everything(seed)

seq_length=7 # 时间步长
input_size=134
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

X, y = get_Data()
X=normalization_x(X)

X_ss, y_en = split_sequences(X, y, seq_length)
train,test= get_split_data(X_ss, y_en, False)

train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, num_workers=0,drop_last=True)
test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True, num_workers=0,drop_last=True)
test_dataloader_one = DataLoader(dataset=test, batch_size=1, shuffle=False, num_workers=0,drop_last=True)
#-------------------add by funing for getdataset---------------------------------------------


#in_channels, out_channels, hidden_size, num_layers, output_size, batch_size, seq_length)
model=Net_classification(input_size,out_channels,hidden_size,num_layers,output_size,batch_size,kernel_size,seq_length,dropput)
criterion=torch.nn.CrossEntropyLoss
optimizer=torch.optim.RMSprop(model.parameters(),lr=lr)
print(model)

opt = Net_classification_optimization(model=model, loss_fn=criterion, optimizer=optimizer)
opt.train(train_loader, n_epochs=network_epoch)
predictions, values = opt.evaluate_for_one(test_dataloader_one, 1, input_size)
y_test_pred,y_test,fitness = opt.format_predictions(predictions, values)

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



