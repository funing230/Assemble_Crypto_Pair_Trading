# univariate mlp example
import math

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from mealpy.evolutionary_based import GA
from mealpy.swarm_based import GWO
from permetrics.regression import RegressionMetric
import tensorflow as tf
import statsmodels.regression.linear_model as rg
import numpy as np
import random
random.seed(7)
np.random.seed(42)
tf.random.set_seed(116)
from GA_util_all_data import print_table,pdmdd,normalize_series,triple_barrier,calculate_mdd,get_mdd
from numpy import array, reshape
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler



class HybridMlp:
    def __init__(self, dataset, pair_train,GA_epoch,pop_size):  #dataset,
        self.dataset = dataset
        self.pair_train = pair_train
        self.GA_epoch = GA_epoch
        self.pop_size = pop_size


    def create_problem(self):

        LB = [1.8,      0.99,      2  , 20]
        UB = [1.01,     0.01,      60  ,90]

        self.problem = {
            "fit_func": self.fitness_function,
            "lb": LB,
            "ub": UB,
            "minmax": "max",
            "log_to": None,
            "obj_weights": [0.20,0.80],
            "save_population": False,
        }

    def decode_solution(self,solution):
        a = solution[0]
        b = solution[1]
        k = int(solution[2])
        window= int(solution[3])


        return {
            "a": a,
            "b": b,
            "k": k,
            "window":window
        }
    def fitness_function(self, solution):

        structure = self.decode_solution(solution)

        # 预测数据步长为1,一个预测一个，1->1
        a = structure["a"]
        b = structure["b"]
        k=  structure["k"]
        window = structure["window"]

        # tests=tests[1:]  #*****************************************************************************************************
        tests = self.dataset.dropna()
        ftestsig2 = 0.0
        ftestsig2a = []
        # add by funing 0323 for test triple_barrier

        z_score = (self.pair_train - self.pair_train.rolling(window=window, min_periods=1).mean()) / self.pair_train.rolling(window=window, min_periods=1).std()

        z_score = z_score.dropna()

        z_score_ret = triple_barrier(z_score, a, b, k)

        z_score_singel = z_score_ret['triple_barrier_signal']

        ftestsig2a = z_score_singel

        tests.insert(len(tests.columns), 'ftestsig2', ftestsig2a)
        tests.insert(len(tests.columns), 'rbtc_ret', rbtc_ret)
        tests.insert(len(tests.columns), 'reth_ret', reth_ret)

        # 3.3.6 Trading Strategy Signals, without commission/exchange fee

        port_out = 0.0
        port_outa = []

        for i in range(0, len(tests.index)):
            if tests.at[tests.index[i], 'ftestsig2'] == -2:
                port_out = tests.at[tests.index[i], 'rbtc_ret']
            elif tests.at[tests.index[i], 'ftestsig2'] == -1:
                port_out = tests.at[tests.index[i], 'reth_ret']
            elif tests.at[tests.index[i], 'ftestsig2'] == 2:
                port_out = tests.at[tests.index[i], 'rbtc_ret']
            elif tests.at[tests.index[i], 'ftestsig2'] == 1:
                port_out = tests.at[tests.index[i], 'reth_ret']
            else:
                port_out = tests.at[tests.index[i], 'rbtc_ret']
            port_outa.append(port_out)
        tests.insert(len(tests.columns), 'port_out', port_outa)
        tests = tests.fillna(method='ffill')

        pt_out = np.exp(np.log1p(tests['port_out']).cumsum()) - 1  # pair trading return
        # pt_out = pt_out.iloc[3:]
        _,_,MDD=get_mdd(pt_out)
        print("------FINALLY---------------------------------------")
        print("Return : " + str(np.round(pt_out.iloc[-1], 4)))
        print("Standard Deviation : " + str(np.round(np.std(pt_out), 4)))  # mean_absolute_percentage_error
        print("Sharpe Ratio (Rf=0%) : " + str(np.round(pt_out.iloc[-1] / (np.std(pt_out)), 4)))
        print("Max Drawdown: " + str(np.round(MDD, 4)))  #  calculate_mdd(pt_out)
        print('++++++++++++++++++++++++++++++++++++++')
        print("a : " + str(a))
        print("b : " + str(b))
        print("k : " + str(k))
        print("window : " + str(window))
        print('-------------------------------------')

        fitness=[np.round(pt_out.iloc[-1], 4), np.round(MDD+10, 4)]

        # np.savetxt("./predict_value/"+ str(fitness) +"_GA_Predict_DS3.csv", np.numpy(predict), fmt='%d')

        return fitness
    def training(self):
        self.create_problem()
        self.optimizer = GA.BaseGA(self.problem, GAepoch=self.GA_epoch,pop_size=self.pop_size, pc=0.8, pm=0.3)
        self.solution, self.best_fit = self.optimizer.solve()

    def best_fitness(self):
        Return, MDD = self.model.best_fit
        print('-------------------------------------')
        print("Return : " + str(Return))
        print("Max Drawdown: " + str(MDD))
        print('-------------------------------------')

    def best_model(self):
        structure = self.decode_solution(self.solution)
        print("------FINALLY-----------------------------------")
        print("a", structure["a"], )
        print("b", structure["b"], )
        print("k",structure["k"],)
        print("window", structure["window"])




totaldataset_file_path = '../totaldataset_df_BTC.csv'
BTC_df = pd.read_csv(totaldataset_file_path, parse_dates=[0], index_col=0)
totaldataset_file_path = '../totaldataset_df_ETH.csv'
ETH_df = pd.read_csv(totaldataset_file_path, parse_dates=[0], index_col=0)

pair= pd.concat([BTC_df['close'],ETH_df['close']], ignore_index=True,axis=1)

pair_ret=normalize_series(pair)

#remove first row with NAs
pair_ret=pair_ret.tail(len(pair_ret)-1)
pair_ret.columns = ['BTC_RET','ETH_RET']
# Then we split our Bitcoin Returns and Ethereum Returns into the training data set and the testing data set.
# After which we concatenated the tests into the tests data set.

#3.1.3 split into train and validation/testing
btc_R_train =  pair_ret['BTC_RET'][:809]
btc_R_test =   pair_ret['BTC_RET'][:809]
eth_R_train = pair_ret['ETH_RET'][:809]
eth_R_test =  pair_ret['ETH_RET'][:809]



#3.3   z_score=(pair_spread-spread_mean)/spread_sd

pair_train= btc_R_test - rg.OLS(btc_R_train, eth_R_train).fit().params[0] * eth_R_test
# BTC_ETH Rolling Spread Z-Score Calculation


#3.3.4  getting equalent values of the returns, introducing return on returns to build signals
rbtc_ret= pair_ret['BTC_RET'][:809] # picking the last 120 transactions on Bitcoin  for our trading
reth_ret= pair_ret['ETH_RET'][:809] # picking the last 120 transactions on Ethereum  for our trading

tests= pd.concat([btc_R_test ,eth_R_test], ignore_index=False,axis=1)

GA_epoch=100
GA_pop_size=20


## Create hybrid model
model = HybridMlp(tests,pair_train,GA_epoch,GA_pop_size) #dataset,

model.training()

model.best_model()

model.best_fitness()
