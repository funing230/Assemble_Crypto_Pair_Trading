# univariate mlp example
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# from timeseries_util import split_sequences,LSTM,fitness_calculate,MyDataset
from GA_Assemble_util import Net,CNNLSTMOptimization,MyDataset,split_sequences,seed_everything,get_data ,Normalizer
from mealpy.evolutionary_based import GA
from mealpy.swarm_based import GWO
from permetrics.regression import RegressionMetric
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import warnings


warnings.filterwarnings("ignore")
seed = 42
seed_everything(seed)

class HybridMlp:
    def __init__(self, GA_epoch, pop_size):  #dataset, GA_epoch, pop_size
        self.X_train, self.X_test, self.Y_train, self.Y_test = None, None, None, None
        # self.n_hidden_nodes = n_hidden_nodes
        self.GA_epoch = GA_epoch
        self.pop_size = pop_size
        self.model, self.problem, self.optimizer, self.solution, self.best_fit = None, None, None, None, None
        self.n_dims, self.n_inputs = None, None
        self.data=None
        # self.dataset=dataset
        self.term_dict = None

    def create_problem(self):
        # LABEL ENCODER
        OPT_ENCODER = LabelEncoder()
        OPT_ENCODER.fit(["SGD", "Adam", "RMSprop", "Rprop", "Adamax", "Adagrad"])

        LOSS_ENCODER = LabelEncoder()
        LOSS_ENCODER.fit(['MSELoss', 'L1Loss', 'SmoothL1Loss'])  #'MSELoss', 'L1Loss', 'SmoothL1Loss'

        DATA = {}
        DATA["OPT_ENCODER"] = OPT_ENCODER
        # DATA["STEP_ENCODER"] = STEP_ENCODER
        DATA["LOSS_ENCODER"] = LOSS_ENCODER
        #
        # LB = [1,    1,     0,      0.001,    1,      1,     0.05,   0,      0]
        # UB = [2,    2.99,  3.99,   0.01,     7.99,   7.99,  0.5,    4.99,   2.99]

        LB = [1,     1,     0,     0.001,   0,      0,   1,    1,   0.1, 1,  32 ,  1]
        UB = [6,     5,  5.99,    0.03,     4,   2.99,   6,    3,   0.8, 10, 128,  5]

        self.problem = {
            "fit_func": self.fitness_function,
            "lb": LB,
            "ub": UB,
            "minmax": "min",
            "log_to": None,
            # "obj_weights": [0.3,0.2, 0.2,0.1,0.195,0.005],
            "save_population": False,
            # "data": DATA,
        }
        self.term_dict = {  # When creating this object, it will override the default epoch you define in your model
            "mode": "MG",
            "quantity":300  # 1000 epochs
        }
        self.data = DATA
        return self.problem


    def decode_solution(self,solution, data):
        batch_size = 2 ** int(solution[0])
        network_epoch = 50 * int(solution[1])
        opt_integer = int(solution[2])
        opt = data["OPT_ENCODER"].inverse_transform([opt_integer])[0]
        learning_rate = solution[3]
        n_steps = 2**int(solution[4])
        loss_integer = int(solution[5])
        loss=data["LOSS_ENCODER"].inverse_transform([loss_integer])[0]
        hidden_dim = 16*int(solution[6])
        num_layers= int(solution[7])
        dropout = solution[8]
        out_channels = int(solution[9])
        hidden_size = int(solution[10])
        kernel_size = int(solution[11])

        return {
            "batch_size": batch_size,
            "network_epoch": network_epoch,
            "opt": opt,
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "loss": loss,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "out_channels": out_channels,
            "hidden_size": hidden_size,
            "kernel_size": kernel_size

        }

    def fitness_function(self, solution):

        structure = self.decode_solution(solution, self.data)

        n_steps = int(structure["n_steps"])

        totaldataset_file_path = '../Dataset/totaldataset_df_new.csv'
        totaldataset_df = pd.read_csv(totaldataset_file_path, parse_dates=[0], index_col=0)

        X, y = totaldataset_df.drop(columns=['close']), totaldataset_df['close']

        # Normalization------------------------------Data Processing----------------------
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler_y = MinMaxScaler(feature_range=(-1, 1))
        # X = X.values
        y = np.array(y).reshape(-1, 1)
        X_total = scaler.fit_transform(X)  # ss.fit_transform(X)  normalize(X)
        y_total = scaler_y.fit_transform(y)

        # window=30
        # scaler = Normalizer()
        # scaler_y = Normalizer()
        # X_total=scaler.fit_transform(X,window=window)
        # print(X_total.isnull().sum())
        # y_total=scaler_y.fit_transform(pd.DataFrame(y),window=window)
        # print(y_total.isnull().sum())
        # END---------------Normalization------------------------------Data Processing----------------------

        # Cut dataset by time step[1,2,3,4,5,6.......30]=====>[31]
        X_ss, y_en = split_sequences(X_total, y_total, n_steps)
        # print(X_ss)
        # print("Create time step :",totaldataset_df.shape)
        # total_samples = len(X)
        train, test=get_data(X_ss, y_en,False)

        #Model parameter
        input_dim = 66   #total feature number
        output_dim = 1

        batch_size=int(structure["batch_size"])
        hidden_dim = int(structure["hidden_dim"])
        num_layers = int(structure["num_layers"])
        dropout = structure["dropout"]
        out_channels = structure["out_channels"]
        hidden_size = structure["hidden_size"]

        kernel_size = structure["kernel_size"] if (n_steps-1)>structure["kernel_size"] else 1

        train_dataloader = DataLoader(dataset=train,batch_size=batch_size,shuffle=True,num_workers=0)
        test_dataloader = DataLoader(dataset=test,batch_size=batch_size,shuffle=True,num_workers=0)
        test_dataloader_one = DataLoader(dataset=test, batch_size=1, shuffle=False, num_workers=0)

        # Model construction      out_channels  hidden_size kernel_size
        model = Net(input_dim,out_channels,hidden_size,num_layers,output_dim,batch_size,kernel_size,n_steps,dropout)
        loss_fn = getattr(torch.nn, structure["loss"])(model.parameters(), reduction ="sum")
        optimiser = getattr(torch.optim, structure["opt"])(model.parameters(), lr=structure["learning_rate"])
        # Compile model
# -----------------------------------------------------------------for Traing
        network_epoch = int(structure["network_epoch"])
        opt = CNNLSTMOptimization(model=model,loss_fn=loss_fn, optimizer=optimiser)
        opt.train(train_dataloader, n_epochs=network_epoch)
        predictions, values = opt.evaluate_for_one(test_dataloader_one,1,input_dim)
        y_test_pred,y_test,fitness = opt.format_predictions(predictions, values,scaler_y)
        # save model weights funing
        torch.save(model, "./model/" + str(fitness)+ "_weight.pth")

        # file =str(fitness) #str(self.best_fit[0])
        # file = file.replace('[', '').replace(']', '')
        # location = "./model/" + file + "_weight.pth"  # DS3_0.8628637164174774
        # load_module = torch.load(location)
        # print(model)
        print("---batch_size---:",batch_size,"---n_steps---:",n_steps,"---network_epoch---:",network_epoch)
        print("---hidden_dim---:",hidden_dim,"---dropout---:",dropout,"---num_layers---:",num_layers)
        print("---loss---:", structure["loss"],"---opt---:", structure["opt"],"---learning_rate---:", structure["learning_rate"])
        print("----FITNESS-----------RMSE----",fitness,"----------")
        return fitness

    def prediction_value(self,solution):
        file =str(str(self.best_fit[0])) #str(self.best_fit[0])
        # file = file.replace('[', '').replace(']', '')
        location = "./model/" + file + "_weight.pth"  # DS3_0.8628637164174774
        load_module = torch.load(location)
        print(load_module)
        print()

    def training(self):
        self.problem=self.create_problem()
        model =GA.BaseGA(epoch=self.GA_epoch, pop_size=self.pop_size, pc=0.9, pm=0.01)
        self.solution, self.best_fit = model.solve(self.problem)

    def best_fitness(self):
        rmspe_score= self.best_fit
        print('-------------------------------------')
        # mse, mape, smape, mpe,rmspe=model.best_fit
        # print('-------------------------------------')
        print("rmspe_score : " + str(rmspe_score))  # root_mean_squared_error
        print('-------------------------------------')

    def best_model(self):  #    DS3_0.8628637164174774
        structure = self.decode_solution(self.solution, self.data)
        return structure


import logging
def initLogging(logFilename):
  """Init for logging
  """
  logging.basicConfig(
                    level    = logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H:%M',
                    filename = logFilename,
                    filemode = 'w');
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
  console.setFormatter(formatter)
  logging.getLogger('').addHandler(console)

initLogging('test_infor.log')
logging.info('just play')

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

sys.stdout = Logger("test.log", sys.stdout)
sys.stderr = Logger("test_error.log", sys.stderr)		# redirect std err, if necessary




GA_epoch=200
GA_pop_size=20

## Create hybrid model
GAmodel = HybridMlp(GA_epoch,GA_pop_size) #dataset,
GAmodel.training()
GAmodel.best_model()
GAmodel.best_fitness()

