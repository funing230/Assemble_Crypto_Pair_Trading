# univariate mlp example
from sklearn.preprocessing import LabelEncoder
# from timeseries_util import split_sequences,LSTM,fitness_calculate,MyDataset
from CNNLSTM_Baseline.GA_util import Optimization, split_sequences,seed_everything,get_data
from mealpy.evolutionary_based import GA
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
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
        # STEP_ENCODER = LabelEncoder()
        # STEP_ENCODER.fit(['1', '2', '3', '4', '5',])
        LOSS_ENCODER = LabelEncoder()
        LOSS_ENCODER.fit(['MSELoss', 'L1Loss', 'SmoothL1Loss'])  #'MSELoss', 'L1Loss', 'SmoothL1Loss'

        DATA = {}
        DATA["OPT_ENCODER"] = OPT_ENCODER
        # DATA["STEP_ENCODER"] = STEP_ENCODER
        DATA["LOSS_ENCODER"] = LOSS_ENCODER
        #
        # LB = [1,    1,     0,      0.001,    1,      1,     0.05,   0,      0]
        # UB = [2,    2.99,  3.99,   0.01,     7.99,   7.99,  0.5,    4.99,   2.99]

        LB = [1,     1,     0,     0.001,   0,      0,   1,    1,   0.1, ]
        UB = [6,     5,  5.99,    0.03,     4,   2.99,   6,    3,   0.8, ]

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
        network_epoch = 500 * int(solution[1])
        opt_integer = int(solution[2])
        opt = data["OPT_ENCODER"].inverse_transform([opt_integer])[0]
        learning_rate = solution[3]
        n_steps = 2**int(solution[4])
        loss_integer = int(solution[5])
        loss=data["LOSS_ENCODER"].inverse_transform([loss_integer])[0]

        hidden_dim = 16*int(solution[6])

        num_layers= int(solution[7])

        dropout = solution[8]

        return {
            "batch_size": batch_size,
            "network_epoch": network_epoch,
            "opt": opt,
            "learning_rate": learning_rate,
            "n_steps":n_steps,
            "loss": loss,
            "hidden_dim":hidden_dim,
            "num_layers":num_layers,
            "dropout":dropout
        }

    def fitness_function(self, solution):

        return ''#fitness

    def prediction_value(self):
        # structure = self.decode_solution(solution, self.data)
        file ="2492.1879945140577"#str(str(self.best_fit[0])) #str(self.best_fit[0])
        # file = file.replace('[', '').replace(']', '')
        location = "./model/" + file + "_weight.pth"  #

        n_steps =4 #int(structure["n_steps"])
        totaldataset_file_path = '../Dataset/total_dataset.csv'
        totaldataset_df = pd.read_csv(totaldataset_file_path, parse_dates=[0], index_col=0)
        X, y = totaldataset_df.drop(columns=['close']), totaldataset_df['close']

        # Normalization------------------------------Data Processing----------------------
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler_y = MinMaxScaler(feature_range=(-1, 1))
        X = X.values
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
        total_samples = len(X)
        train, test=get_data(X_ss, y_en,total_samples)

        #Model parameter
        input_dim = 32   #total feature number
        output_dim = 1
        batch_size=2#int(structure["batch_size"])

        # hidden_dim = 64#int(structure["hidden_dim"])
        # num_layers = 2#int(structure["num_layers"])
        # dropout =0.4305504476133646#structure["dropout"]

        train_dataloader = DataLoader(dataset=train,batch_size=batch_size,shuffle=False,num_workers=0)
        test_dataloader = DataLoader(dataset=test,batch_size=batch_size,shuffle=False,num_workers=0)
        test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

        # Model construction
        load_module = torch.load(location)
        loss_fn = getattr(torch.nn,  "L1Loss")(load_module.parameters(), reduction ="sum")
        optimiser = getattr(torch.optim, "Adagrad")(load_module.parameters(), lr=0.0001)

        opt = Optimization(model=load_module, loss_fn=loss_fn, optimizer=optimiser)

        predictions, values = opt.evaluate_for_one(test_loader_one, batch_size=1, n_features=input_dim)

        fitness = opt.format_predictions(predictions, values, scaler_y)
        # print('------FITNESS-------',fitness)

        vals = np.concatenate(values, axis=0).ravel()
        preds = np.concatenate(predictions, axis=0).ravel()
        y_test_pred = scaler_y.inverse_transform(preds.reshape(-1, 1))
        y_test = scaler_y.inverse_transform(vals.reshape(-1, 1))

        np.savetxt("GA_CNNLSTM_ytest_pred.csv", y_test_pred, delimiter=',')
        np.savetxt("GA_CNNLSTM_ytest.csv", y_test, delimiter=',')

        # plot baseline and predictions
        plt.figure(figsize=(15, 8))
        plt.title('Testing')
        plt.plot(y_test_pred, label="Preds")
        plt.plot(y_test, label="Data")
        plt.legend()
        plt.show()

        # load_module = torch.load(location)
        print(load_module)


    def training(self):
        self.problem=self.create_problem()
        model =GA.BaseGA(epoch=self.GA_epoch, pop_size=self.pop_size, pc=0.9, pm=0.01)
        self.solution, self.best_fit = model.solve(self.problem)

    def best_fitness(self):
        rmspe_score= self.best_fit
        print('-------------------------------------')
        # mse, mape, smape, mpe,rmspe=model.best_fit
        # print('-------------------------------------')
        # print("rmspe_score : " + str(rmspe_score))  # root_mean_squared_error
        print('-------------------------------------')

    def best_model(self):  #    DS3_0.8628637164174774
        structure = self.decode_solution(self.solution, self.data)
        return structure



GA_epoch=200
GA_pop_size=20

## Create hybrid model
GAmodel = HybridMlp(GA_epoch,GA_pop_size) #dataset,
GAmodel.prediction_value()




