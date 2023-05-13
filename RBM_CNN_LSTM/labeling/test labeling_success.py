#import packages
import pandas as pd
import numpy as np
import glob,os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sn

np.random.seed(0) # for reproducibility

from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_pacf
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.vector_ar.vecm import coint_johansen
#setting figure size
from matplotlib.pylab import rcParams
from GA_util_all_data import print_table,pdmdd,normalize_series,triple_barrier,calculate_mdd,get_mdd
rcParams['figure.figsize'] = 20,10



#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

from datetime import datetime
import statsmodels.regression.linear_model as rg

import pprint
import warnings
warnings.filterwarnings("ignore")
#3.1 calling the individual Cryptocurrencies

# BTC = yf.download('BTC-USD', start=datetime(2018, 10, 3), end=datetime(2021, 12, 1))
# ETH = yf.download('ETH-USD', start=datetime(2018, 10, 3), end=datetime(2021, 12, 1))


totaldataset_file_path = '../totaldataset_df_BTC.csv'
BTC_df = pd.read_csv(totaldataset_file_path, parse_dates=[0], index_col=0)
totaldataset_file_path = '../totaldataset_df_ETH.csv'
ETH_df = pd.read_csv(totaldataset_file_path, parse_dates=[0], index_col=0)



pair= pd.concat([BTC_df['close'],ETH_df['close']], ignore_index=True,axis=1)

#3.1.1 Normalizing the datafrme

def normalize_series(pair):
    #take tail to drop head NA
    return pair.pct_change(1).dropna()
pair_ret=normalize_series(pair)

#remove first row with NAs
pair_ret=pair_ret.tail(len(pair_ret)-1)
pair_ret.columns = ['BTC_RET','ETH_RET']

# Then we split our Bitcoin Returns and Ethereum Returns into the training data set and the testing data set.
# After which we concatenated the tests into the tests data set.

#3.1.3 split into train and validation/testing

btc_R_train =  pair_ret['BTC_RET'][:809]
btc_R_test =   pair_ret['BTC_RET'][809:]
eth_R_train = pair_ret['ETH_RET'][:809]
eth_R_test =  pair_ret['ETH_RET'][809:]
tests= pd.concat([btc_R_test ,eth_R_test], ignore_index=False,axis=1)

#3.1.4 the Spread
pair_spread= btc_R_test - rg.OLS(btc_R_train, eth_R_train).fit().params[0] * eth_R_test
#3.2  calculating for beta (Hedge)
beta= rg.OLS(btc_R_train, eth_R_train).fit().params[0]
#3.2.1 Pairs Spread plot

#3.2.7 Spread Mean and Standard dev
spread_mean= pair_spread.mean()
spread_sd= pair_spread.std()

print('the mean of the spread is', spread_mean)
print('the Standard Dev of the Spread is',spread_sd)

#3.3   z_score=(pair_spread-spread_mean)/spread_sd
window= 52
pair_train= btc_R_test - rg.OLS(btc_R_train, eth_R_train).fit().params[0] * eth_R_test

# BTC_ETH Rolling Spread Z-Score Calculation
z_score = (pair_train - pair_train.rolling(window=window,min_periods=1).mean()) / pair_train.rolling(window=window,min_periods=1).std()


z_score=z_score.dropna()
z_score_ret=triple_barrier(z_score, 1.1475588244841508, 0.09351398897506247, 3)
z_score_singel=z_score_ret['triple_barrier_signal'].tail(342)

#3.3.2 thresholds
# print(z_score.rolling(window=2,min_periods=1).mean())
z_score_mean=z_score.rolling(window=2,min_periods=1).mean()
# z_score_mean=z_score_mean[1:]
# print(z_score.rolling(window=2,min_periods=1).std())
z_score_std=z_score.rolling(window=2,min_periods=1).std()
z_score_std.dropna()

up_th = (z_score.rolling(window=2).mean())+(z_score.rolling(window=2).std()*2) # upper threshold
lw_th = (z_score.rolling(window=2).mean())-(z_score.rolling(window=2).std()*2) # lower threshold
# up_th = (z_score_mean)+(z_score_std*2) # upper threshold
# lw_th = (z_score_mean)-(z_score_std*2) # lower threshold
up_th=up_th.dropna()
lw_th=lw_th.dropna()
up_lw= pd.concat([up_th,lw_th, btc_R_test ,eth_R_test,z_score,z_score.rolling(window=2).mean(),z_score.rolling(window=2).std()*2], ignore_index=True,axis=1)
up_lw.columns = [ 'up_th','lw_th','btc_R_test','eth_R_test','z_score','mean(window=2)','deviation(window=2)']
up_lw.dropna()

#3.3.3 Z-score Plot with threshold
plt.figure(figsize=(16,8))
plt.rcParams.update({'font.size':10})
plt.xticks(rotation=45)
ax = plt.axes()
ax.xaxis.set_major_locator(plt.MaxNLocator(20))
plt.plot(z_score,color='blue',label='Z-score')
plt.plot(up_th,color='red',linestyle='--', label='upperLimit')
plt.plot(lw_th,color='brown',linestyle='--', label='lowerLimit')
plt.suptitle('Z-score')
ax.axhline(z_score.mean(), color='orange')
ax.grid(True)
plt.xlabel("Date")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.legend(loc='best')
plt.show()

#3.3.4  getting equalent values of the returns, introducing return on returns to build signals
rbtc_ret= pair_ret['BTC_RET'].tail(347) # picking the last 120 transactions on Bitcoin  for our trading
reth_ret= pair_ret['ETH_RET'].tail(347) # picking the last 120 transactions on Ethereum  for our trading
#return on returns
rrbtc=(pair_ret['BTC_RET'].pct_change(1).dropna()).pct_change(1).dropna().tail(347)#return on returns for bitcoin

rreth=(pair_ret['ETH_RET'].pct_change(1).dropna()).pct_change(1).dropna().tail(347)#return on returns for ethereum

trade_dir= pd.DataFrame(rbtc_ret)

trade_dir.insert(len(trade_dir.columns), 'rbtc_ret(-1)', rbtc_ret.shift(1))
trade_dir.insert(len(trade_dir.columns), 'rbtc_ret(-2)', rbtc_ret.shift(2))
trade_dir.insert(len(trade_dir.columns), 'reth_ret(-1)', reth_ret.shift(1))
trade_dir.insert(len(trade_dir.columns), 'reth_ret(-2)', reth_ret.shift(2))
trade_dir.insert(len(trade_dir.columns), 'rrbtc(-1)', rrbtc.shift(1))
trade_dir.insert(len(trade_dir.columns), 'rrbtc(-2)', rrbtc.shift(2))
trade_dir.insert(len(trade_dir.columns), 'rreth(-1)', rreth.shift(1))
trade_dir.insert(len(trade_dir.columns), 'rreth(-2)', rreth.shift(2))
trade_dirsig2 = 0.0
trade_dirsig2a = []


# trade_dir=trade_dir[1:]
#2022.9.7 add for generate signs
plt.figure(figsize=(16, 8))
plt.rcParams.update({'font.size': 10})
plt.xticks(rotation=45)
ax = plt.axes()
ax.xaxis.set_major_locator(plt.MaxNLocator(20))
plt.plot(trade_dir['rrbtc(-1)'], color='blue', label='Z-rrbtc(-1)')
plt.plot(trade_dir['rrbtc(-2)'], color='red', linestyle='--', label='rrbtc(-2)')
plt.plot(rreth, color='brown', linestyle='--', label='rreth')
plt.suptitle('Z-score')
ax.axhline(z_score.mean(), color='orange')
ax.grid(True)
plt.xlabel("Date")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.legend(loc='best')
plt.show()


trade_dir=trade_dir.dropna()

for i in range(0,len(trade_dir.index)):
    if trade_dir.at[trade_dir.index[i], 'rrbtc(-2)'] > (rreth[trade_dir.index[i]]) and trade_dir.at[trade_dir.index[i], 'rrbtc(-1)'] < (rreth[trade_dir.index[i]]):
        trade_dirsig2 = 2
    elif trade_dir.at[trade_dir.index[i], 'rrbtc(-2)'] < (rreth[trade_dir.index[i]]) and trade_dir.at[trade_dir.index[i], 'rrbtc(-1)'] > (rreth[trade_dir.index[i]]):
        trade_dirsig2 = -2
    elif trade_dir.at[trade_dir.index[i], 'rreth(-2)'] > (rrbtc[trade_dir.index[i]]) and trade_dir.at[trade_dir.index[i], 'rreth(-1)'] < (rrbtc[trade_dir.index[i]]):
        trade_dirsig2 = 1
    elif trade_dir.at[trade_dir.index[i], 'rreth(-2)'] < (rrbtc[trade_dir.index[i]]) and trade_dir.at[trade_dir.index[i], 'rreth(-1)'] > (rrbtc[trade_dir.index[i]]):
        trade_dirsig2 = -1
    else:
        trade_dirsig2 = 0.0
    trade_dirsig2a.append(trade_dirsig2)

trade_dir.insert(len(trade_dir.columns), 'trade_dirsig2', trade_dirsig2a)


#3.3.5. BTC_ETH Trading Strategy Signals
tests.insert(len(tests.columns), 'z_score', z_score)
tests.insert(len(tests.columns), 'z_score(-1)', z_score.shift(1))
tests.insert(len(tests.columns), 'z_score(-2)', z_score.shift(2))
tests.insert(len(tests.columns), 'trade_dir', trade_dir['trade_dirsig2'])

# tests=tests[1:]  #*****************************************************************************************************
tests=tests.dropna()

plt.figure(figsize=(30,15))
plt.rcParams.update({'font.size':12})
plt.xticks(rotation=45)
ax = plt.axes()
ax.xaxis.set_major_locator(plt.MaxNLocator(20))
plt.plot(z_score,color='blue',label='Z-score')
plt.plot(up_th,color='red',linestyle='--', label='upperLimit')
plt.plot(lw_th,color='brown',linestyle='--', label='lowerLimit')
plt.suptitle('Z-score')
ax.axhline(z_score.mean(), color='orange')
ax.grid(True)
plt.xlabel("Date")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.legend(loc='best')
plt.show()



ftestsig2 = 0.0
ftestsig2a = []
for i in range(0,len(tests.index)):
    if tests.at[tests.index[i], 'z_score(-2)'] > (-1*up_th[tests.index[i]]) and tests.at[tests.index[i], 'z_score(-1)'] < (-1*up_th[tests.index[i]]):
        ftestsig2 = 1
    elif tests.at[tests.index[i], 'z_score(-2)'] < (-1*lw_th[tests.index[i]]) and tests.at[tests.index[i], 'z_score(-1)'] > (-1*lw_th[tests.index[i]]):
        ftestsig2 = -2
    elif tests.at[tests.index[i], 'z_score(-2)'] < (-1*up_th[tests.index[i]]) and tests.at[tests.index[i], 'z_score(-1)'] > (-1*up_th[tests.index[i]]):
        ftestsig2 = -1
    elif tests.at[tests.index[i], 'z_score(-2)'] > (-1*lw_th[tests.index[i]]) and tests.at[tests.index[i], 'z_score(-1)'] < (-1*lw_th[tests.index[i]]):
        ftestsig2 = 2
    elif tests.at[tests.index[i], 'z_score(-2)'] < up_th[tests.index[i]] and tests.at[tests.index[i], 'z_score(-1)'] > up_th[tests.index[i]]:
        ftestsig2 = -1
    elif tests.at[tests.index[i], 'z_score(-2)'] > up_th[tests.index[i]] and tests.at[tests.index[i], 'z_score(-1)'] < up_th[tests.index[i]]:
        ftestsig2 = 1
    elif tests.at[tests.index[i], 'z_score(-2)'] > lw_th[tests.index[i]] and tests.at[tests.index[i], 'z_score(-1)'] < lw_th[tests.index[i]]:
        ftestsig2 = 2
    elif tests.at[tests.index[i], 'z_score(-2)'] < lw_th[tests.index[i]] and tests.at[tests.index[i], 'z_score(-1)'] > lw_th[tests.index[i]]:
        ftestsig2 = -2
    elif tests.at[tests.index[i], 'trade_dir'] ==1:
        ftestsig2 = 2
    elif tests.at[tests.index[i], 'trade_dir'] == -1:
        ftestsig2 = -2
    elif tests.at[tests.index[i], 'trade_dir'] == 2:
        ftestsig2 = 1
    elif tests.at[tests.index[i], 'trade_dir']== -2:
        ftestsig2 = -1
    else:
        ftestsig2 = 0.0
    ftestsig2a.append(ftestsig2)


#add by funing 0323 for test triple_barrier
# ftestsig2a=z_score_singel
tests.insert(len(tests.columns), 'ftestsig2', ftestsig2a)
tests.insert(len(tests.columns), 'z_score_singel', z_score_singel)



df = z_score_singel.to_frame().reset_index()
df.columns = ['date','z_score_singel']
df.to_csv("z_score_singel.csv", index=False)

print('== BTC_ETH Trading Strategy Signals ==')
print('')
print(tests.loc['2021-11-20':, ['z_score', 'ftestsig2']])


tests.insert(len(tests.columns), 'rbtc_ret', rbtc_ret)
tests.insert(len(tests.columns), 'reth_ret', reth_ret)

#3.3.6 Trading Strategy Signals, without commission/exchange fee

port_out = 0.0
port_outa = []

# If the value of 'ftestsig2' is -2, 'port_out' is assigned the value in the column 'rbtc_ret'.
# If the value of 'ftestsig2' is -1, 'port_out' is assigned the value in the column 'reth_ret'.
# If the value of 'ftestsig2' is 2, 'port_out' is assigned the value in the column 'rbtc_ret'.
# If the value of 'ftestsig2' is 1, 'port_out' is assigned the value in the column 'reth_ret'.

for i in range(0,len(tests.index)):
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


port_out_z_score_singel = 0.0
port_outa_z_score_singel = []

# If the value of 'z_score_singel' is 2, 'port_out' is assigned the value in the column 'rbtc_ret'.
# If the value of 'z_score_singel' is 1, 'port_out' is assigned the value in the column 'reth_ret'.

for i in range(0,len(tests.index)):
    if tests.at[tests.index[i], 'z_score_singel'] == -2:
        port_out_z_score_singel = tests.at[tests.index[i], 'rbtc_ret']
    elif tests.at[tests.index[i], 'z_score_singel'] == -1:
        port_out_z_score_singel = tests.at[tests.index[i], 'reth_ret']
    elif tests.at[tests.index[i], 'z_score_singel'] == 2:
        port_out_z_score_singel = tests.at[tests.index[i], 'rbtc_ret']
    elif tests.at[tests.index[i], 'z_score_singel'] == 1:
        port_out_z_score_singel = tests.at[tests.index[i], 'reth_ret']
    else:
        port_out_z_score_singel = tests.at[tests.index[i], 'rbtc_ret']
    port_outa_z_score_singel.append(port_out_z_score_singel)
tests.insert(len(tests.columns), 'port_outa_z_score_singel', port_outa_z_score_singel)
tests = tests.fillna(method='ffill')





port_outc = 0.0
port_outca = []
# with 5% commission
com =0.05

for i in range(0,len(tests.index)):
    if tests.at[tests.index[i], 'ftestsig2'] == -2:
        port_outc = ((tests.at[tests.index[i], 'rbtc_ret']) -abs(com*((tests.at[tests.index[i], 'rbtc_ret'])-(tests.at[tests.index[i], 'reth_ret']))))
    elif tests.at[tests.index[i], 'ftestsig2'] == -1:
        port_outc = ((tests.at[tests.index[i], 'reth_ret']) -abs(com*((tests.at[tests.index[i], 'rbtc_ret'])-(tests.at[tests.index[i], 'reth_ret']))))
    elif tests.at[tests.index[i], 'ftestsig2'] == 2:
        port_outc = ((tests.at[tests.index[i], 'rbtc_ret']) -abs(com*((tests.at[tests.index[i], 'rbtc_ret'])-(tests.at[tests.index[i], 'reth_ret']))))
    elif tests.at[tests.index[i], 'ftestsig2'] == 1:
        port_outc = ((tests.at[tests.index[i], 'reth_ret']) -abs(com*((tests.at[tests.index[i], 'rbtc_ret'])-(tests.at[tests.index[i], 'reth_ret']))))
    else:
        port_outc = ((tests.at[tests.index[i], 'rbtc_ret']) -abs(com*((tests.at[tests.index[i], 'rbtc_ret'])-(tests.at[tests.index[i], 'reth_ret']))))
    port_outca.append(port_outc)
tests.insert(len(tests.columns), 'port_outc', port_outca)
tests = tests.fillna(method='ffill')

print('== BTC_ETH Trading Strategy Position ==')
print('')
print(tests.loc['2021-11-20':, ['z_score', 'ftestsig2', 'port_out', 'port_outc']])
print('')



#3.4 Output of all 4 Senarios
pt_out = (1 + tests['port_out']).cumprod() - 1#np.exp(np.log1p(tests['port_out']).cumsum())- 1 # pair trading return
pt_out=pt_out.iloc[3:]
port_outa_z_score_singel = (1 + tests['port_outa_z_score_singel']).cumprod() - 1 #np.exp(np.log1p(tests['port_outa_z_score_singel']).cumsum())- 1 # pair trading return
pt_out=pt_out.iloc[3:]
pt_outc = (1 + tests['port_outc']).cumprod() - 1#np.exp(np.log1p(tests['port_outc']).cumsum())- 1 # pair trading with 5% commision  return
pt_outc=pt_outc.iloc[3:]
bh_btc= (1 + tests['rbtc_ret']).cumprod() - 1#np.exp(np.log1p(tests['rbtc_ret']).cumsum())- 1 # buy and Hold Bitcoin
bh_btc=bh_btc.iloc[3:]
bh_eth= (1 + tests['reth_ret']).cumprod() - 1#np.exp(np.log1p(tests['reth_ret']).cumsum())- 1 # buy and Hold Ethereum
bh_eth=bh_eth.iloc[3:]

pt_out_predict = pd.read_csv('predict_pair_trading.csv', index_col=0, parse_dates=True)
pt_out_predict=pt_out_predict.squeeze()



#3.4.1 portfolio returns Chart
plt.figure(figsize=(16,8))
plt.rcParams.update({'font.size':10})
plt.xticks(rotation=45)
ax = plt.axes()
ax.xaxis.set_major_locator(plt.MaxNLocator(20))
plt.plot(pt_out, label='Cumulative return on P-Trading Strategy portfolio',color='b')
plt.plot(port_outa_z_score_singel, label='port_outa_z_score_singel',color='r')
plt.plot(pt_out_predict, label='Cumulative return  for predict',color='y')
# plt.plot(pt_outc, label='Cumulative return on P-Trading Strategy_+5%Cm')
plt.plot(bh_btc, label='Cumulative return on Buy and Hold Bitcoin',color='g')
plt.plot(bh_eth, label='Cumulative return on Buy and Hold Ethereum',color='Purple')
plt.title('BTC_ETH Trading Strategy Cumulative Returns')
plt.xlabel("Date")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.suptitle('BTC_ETH Portfolio Cumulative Returns  (120 days ( 55-days rolling window))')
ax.legend(loc='best')
ax.grid(True)
plt.show()

def pdmdd(prices:pd.Series):
     return (prices - prices.rolling(len(prices), min_periods=1).max()).min()

def calculate_mdd(prices):
    drawdowns = []
    peak = prices[0]

    for price in prices:
        if price > peak:
            peak = price
        drawdown = (peak - price) / peak
        drawdowns.append(drawdown)

    mdd = np.max(drawdowns)
    return mdd

def get_mdd(x):
    """
    MDD(Maximum Draw-Down)
    :return: (mdd rate)
    """
    arr_v = np.array(x)
    peak_lower = np.argmax(np.maximum.accumulate(arr_v) - arr_v)
    peak_upper = np.argmax(arr_v[:peak_lower])
    return (arr_v[peak_lower] - arr_v[peak_upper]) / arr_v[peak_upper]

# Define the function to print the table
def print_table(table):
    # Calculate the width of each column
    col_width = [max(len(str(row[i])) for row in table) for i in range(len(table[0]))]
    # Print the top border of the table
    print("+" + "+".join(['-' * (width + 2) for width in col_width]) + "+")
    # Print the header row
    for i, row in enumerate(table):
        print("|" + "|".join([' {:{}} '.format(str(row[j]), col_width[j]) for j in range(len(row))]) + "|")
        if i == 0:
            # Print the separator between the header and the data rows
            print("+" + "+".join(['=' * (width + 2) for width in col_width]) + "+")
        else:
            # Print the dotted line separator between the data rows
            print("+" + "+".join(['.' * (width + 2) for width in col_width]) + "+")
    # Print the bottom border of the table
    print("+" + "+".join(['-' * (width + 2) for width in col_width]) + "+")



#3.4.2. BTC_ETH Strategy Performance Summary
results2 = [{'0': 'Test:', '1': 'P-Trading Strategy', '2': 'triple barrier labeling', '3': 'Predict Price on Pair Trading','4': 'Buy&Hold Bitcoin','5': 'Buy&Hold Ethereum'},
            {'0': 'Return',
             '1': np.round(pt_out.iloc[-1], 4),
             '2': np.round(port_outa_z_score_singel.iloc[-1], 4),
             '3': np.round(pt_out_predict.iloc[-1], 4),
             '4': np.round(bh_btc.iloc[-1], 4),
             '5': np.round(bh_eth.iloc[-1], 4)},

            {'0': 'Standard Deviation',
             '1': np.round(np.std(pt_out), 4),
             '2': np.round(np.std(port_outa_z_score_singel), 4),
             '3': np.round(np.std(pt_out_predict), 4),
             '4': np.round(np.std(bh_btc), 4),
             '5': np.round(np.std(bh_eth), 4)},

            {'0': 'Sharpe Ratio (Rf=0%)',
             '1': np.round(pt_out.iloc[-1] / (np.std(pt_out)), 4),
             '2': np.round(port_outa_z_score_singel.iloc[-1] / (np.std(port_outa_z_score_singel)), 4),
             '3': np.round(pt_out_predict.iloc[-1] / (np.std(pt_out_predict)), 4),
             '4': np.round(bh_btc.iloc[-1] / (np.std(bh_btc)), 4),
             '5': np.round(bh_eth.iloc[-1] / (np.std(bh_eth)), 4)},

            {'0': 'Max Drawdown',
             '1': np.round(get_mdd(pt_out), 4),
             '2': np.round(get_mdd(port_outa_z_score_singel), 4),
             '3': np.round(get_mdd(pt_out_predict), 4),
             '4': np.round(get_mdd(bh_btc), 4),
             '5': np.round(get_mdd(bh_eth), 4)}
            ]

table2 = pd.DataFrame(results2)

print('')
print('== BTC_ETH Strategy Performance Summary ==')
print('')

# Print the DataFrame as a table
# print(table2.to_string(index=False, header=False))
print_table(table2.values.tolist())


#3.4.2. Analysing Profitabiliy of the Strategy
#
# Max_ret = tests['port_out'].max()
# Min_ret = tests['port_out'].min()
# Min_cret = pt_out.min()
#
# Min_ret_count=len(tests['port_out'].loc[(tests['port_out'] < 0)])
# Max_ret_count =len(tests['port_out'].loc[(tests['port_out'] > 0)])
# Optimal_trade=((Max_ret_count)/(len(tests['port_out'])))*100
#
# print('the Maximum Return per time for this portfolio is',(np.round(Max_ret,2)))
# print('the Minimum Return per time for this portfolio is',(np.round(Min_ret,2)))
# print('the Minimum Cumulative Returns  for this portfolio is',(np.round(Min_cret,2)))
# print('the Profitability of the paired Trading Strategy is',(np.round(Optimal_trade,2)),'%')
# print('trade with the consideration of the hedge ',(np.round(beta,2)),'%','beta','that is 100 value avaliable to bitcoin and ',(np.round(beta*100,0)),' to Ethereum')
