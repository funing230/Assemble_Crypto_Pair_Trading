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
reth_ret= pair_ret['ETH_RET'] [:809]# picking the last 120 transactions on Ethereum  for our trading

tests= pd.concat([btc_R_test ,eth_R_test], ignore_index=False,axis=1)

window= 34

ftestsig2 = 0.0
ftestsig2a = []
# add by funing 0323 for test triple_barrier

z_score = (pair_train - pair_train.rolling(window=window, min_periods=1).mean()) / pair_train.rolling(window=window, min_periods=1).std()

z_score = z_score.dropna()

z_score_ret = triple_barrier(z_score, 1.5941373386425746, 0.01, 5)  #*********************

z_score_singel = z_score_ret['triple_barrier_signal']


df = z_score_singel.to_frame().reset_index()
df.columns = ['date','z_score_singel']
df.to_csv("../z_score_singel.csv", index=False)

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
_, _, MDD = get_mdd(pt_out)



print("------FINALLY---------------------------------------")
print("Return : " + str(np.round(pt_out.iloc[-1], 4)))
print("Standard Deviation : " + str(np.round(np.std(pt_out), 4)))  # mean_absolute_percentage_error
print("Sharpe Ratio (Rf=0%) : " + str(np.round(pt_out.iloc[-1] / (np.std(pt_out)), 4)))
print("Max Drawdown: " + str(np.round(MDD, 4)))  # calculate_mdd(pt_out)
print("window : " + str(window))
print('-------------------------------------')
