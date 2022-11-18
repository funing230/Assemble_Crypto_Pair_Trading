import pandas as pd
from datetime import datetime
import os
import csv
import matplotlib.pyplot as plt
from datetime import date as dt

price_file_path = 'final_ohlcv_data.csv'   #2017.12.31---2021.8.31   “time,high,low,open,volumefrom,volumeto,close”
price_df = pd.read_csv(price_file_path, parse_dates=[0], index_col=0)
price_df.columns = ["high","low","open","volumefrom","volumeto","close"]
price_df=price_df['20181001':'20211201']



volume_file_path = 'final_Volume_data.csv'
# “time,top_tier_volume_quote,top_tier_volume_base,top_tier_volume_total,cccagg_volume_quote,cccagg_volume_base,cccagg_volume_total,total_volume_quote,total_volume_base,total_volume_total”
volume_df = pd.read_csv(volume_file_path, parse_dates=[0], index_col=0)
volume_df.columns = ["top_tier_volume_quote","top_tier_volume_base","top_tier_volume_total",
                    "cccagg_volume_quote","cccagg_volume_base","cccagg_volume_total",
                    "total_volume_quote","total_volume_base","total_volume_total"]
volume_df=volume_df['20181001':'20211201']



transaction_file_path = 'final_Transaction_data.csv'
#“time,zero_balance_addresses_all_time,unique_addresses_all_time,new_addresses,active_addresses,transaction_count,transaction_count_all_time,
# large_transaction_count,average_transaction_value,block_height,hashrate,difficulty,block_time,block_size,current_supply”
transaction_df = pd.read_csv(transaction_file_path, parse_dates=[0], index_col=0)
transaction_df.columns = ["zero_balance_addresses_all_time","unique_addresses_all_time","new_addresses","active_addresses","transaction_count",
                          "transaction_count_all_time","large_transaction_count","average_transaction_value","block_height","hashrate",
                          "difficulty","block_time","block_size","current_supply"]
transaction_df=transaction_df['20181001':'20211201']


socialstats_file_path = 'final_SocialStats_data.csv'
#comments,posts,followers,points
socialstats_df = pd.read_csv(socialstats_file_path, parse_dates=[0], index_col=0)
socialstats_df.columns = ["comments","posts","followers","points"]
socialstats_df=socialstats_df['20181001':'20211201']


total_dataset = pd.concat([price_df,volume_df,transaction_df,socialstats_df],axis=1)

total_dataset.to_csv("total_dataset20221117.csv", index=True)



X, y = total_dataset[["high","low","open","volumefrom","volumeto","top_tier_volume_quote","top_tier_volume_base","top_tier_volume_total",
                    "cccagg_volume_quote","cccagg_volume_base","cccagg_volume_total",
                    "total_volume_quote","total_volume_base","total_volume_total","zero_balance_addresses_all_time","unique_addresses_all_time",
                    "new_addresses","active_addresses","transaction_count","transaction_count_all_time","large_transaction_count",
                    "average_transaction_value","block_height","hashrate",
                    "difficulty","block_time","block_size","current_supply",
                    "comments","posts","followers","points"]], total_dataset['close']
print(X.shape)
print(y.shape)


#Normalization------------------------------Data Processing----------------------
from sklearn.preprocessing import StandardScaler, MinMaxScaler,LabelEncoder

encoder = LabelEncoder()
ss = MinMaxScaler()
X = X.values
X_trans = ss.fit_transform(X)  # ss.fit_transform(X)  normalize(X)
y_trans = encoder.fit_transform(y)

print(type(X_trans))




# X["high","low","open","volumefrom","volumeto","close","top_tier_volume_quote","top_tier_volume_base","top_tier_volume_total",
#                     "cccagg_volume_quote","cccagg_volume_base","cccagg_volume_total",
#                     "total_volume_quote","total_volume_base","total_volume_total","zero_balance_addresses_all_time","unique_addresses_all_time",
#                     "new_addresses","active_addresses","transaction_count","transaction_count_all_time","large_transaction_count",
#                     "average_transaction_value","block_height","hashrate",
#                     "difficulty","block_time","block_size","current_supply",
#                     "comments","posts","followers","points"]



plt.figure(figsize=(16,8))
plt.rcParams.update({'font.size':10})
plt.xticks(rotation=45)
ax = plt.axes()
ax.xaxis.set_major_locator(plt.MaxNLocator(20))
plt.plot(total_dataset)
plt.title('Performance of cryptocurrencies')
plt.legend(total_dataset)
plt.xlabel("Date")
# date=[cal_date(x) for x in total_dataset.index]
ax.grid(True)

plt.show()


pd.DataFrame(X).plot(subplots=True,figsize=(10, 12))
plt.show()

# print(total_dataset.info())
#
#
# total_dataset.index.name='date' #日期为索引列
# #对股票数据的列名重新命名
# total_dataset.columns=["high","low","open","close","volumefrom","volumeto","top_tier_volume_quote","top_tier_volume_base","top_tier_volume_total",
#                     "cccagg_volume_quote","cccagg_volume_base","cccagg_volume_total",
#                     "total_volume_quote","total_volume_base","total_volume_total","zero_balance_addresses_all_time","unique_addresses_all_time",
#                     "new_addresses","active_addresses","transaction_count","transaction_count_all_time","large_transaction_count",
#                     "average_transaction_value","block_height","hashrate",
#                     "difficulty","block_time","block_size","current_supply",
#                     "comments","posts","followers","points"]
# data=total_dataset.loc['2018-10-1':'2021-12-01']  #获取某个时间段内的时间序列数据
# data[['close',"new_addresses","active_addresses","transaction_count","transaction_count_all_time","large_transaction_count"]].plot(secondary_y='transaction_count',grid=True,figsize=(20, 12))
# plt.title('2018-2021 close and volume', fontsize='9')
# plt.show()