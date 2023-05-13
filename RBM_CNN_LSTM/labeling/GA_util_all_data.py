import numpy as np
from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import Dense
# from keras import optimizers
import statsmodels.datasets.co2 as co2
from permetrics.regression import RegressionMetric
import numpy as np
from numpy import array
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
# import pandas_datareader as pdr
from datetime import datetime
import statsmodels.regression.linear_model as rg
import matplotlib.pyplot as plt
import math
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os
import random
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")
np.random.seed(0) # for reproducibility
# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')



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



def triple_barrier(price, ub, lb, max_period):
    def end_price(s):
        return np.append(s[(s / s[0] > ub) | (s / s[0] < lb)], s[-1])[0] / s[0]

    r = np.array(range(max_period))

    def end_time(s):
        return np.append(r[(s / s[0] > ub) | (s / s[0] < lb)], max_period - 1)[0]

    p = price.rolling(max_period).apply(end_price, raw=True).shift(-max_period + 1)
    t = price.rolling(max_period).apply(end_time, raw=True).shift(-max_period + 1)
    t = pd.Series([t.index[int(k + i)] if not math.isnan(k + i) else np.datetime64('NaT')
                   for i, k in enumerate(t)], index=t.index).dropna()

    signal = pd.Series(0, p.index)
    signal.loc[p > ub] = 1
    signal.loc[p < lb] = -1
    ret = pd.DataFrame({'triple_barrier_profit': p, 'triple_barrier_sell_time': t, 'triple_barrier_signal': signal})

    return ret

class Normalizer():
    def __init__(self):
        self.mu = None
        self.sd = None
    def fit_transform(self, x,window):
        self.mu = x.rolling(window=window, min_periods=1).mean()
        self.sd = x.rolling(window=window, min_periods=1).std()
        normalized_x = (x - self.mu)/self.sd
        return normalized_x.dropna()
    def inverse_transform(self, x):
        return ((x*self.sd) + self.mu).dropna()




def pdmdd(prices:pd.Series):
     return (prices - prices.rolling(len(prices), min_periods=1).max()).min()

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



def normalize_series(pair):
    #take tail to drop head NA
    return pair.pct_change(1).dropna()



import numpy as np

def calculate_drawdown(prices):
    max_drawdown = 0
    peak = prices[0]

    for price in prices:
        if price > peak:
            peak = price
        drawdown = (peak - price) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return max_drawdown

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
    :return: (peak_upper, peak_lower, mdd rate)
    """
    arr_v = np.array(x)
    peak_lower = np.argmax(np.maximum.accumulate(arr_v) - arr_v)
    peak_upper = np.argmax(arr_v[:peak_lower])
    return peak_upper, peak_lower, (arr_v[peak_lower] - arr_v[peak_upper]) / arr_v[peak_upper]









