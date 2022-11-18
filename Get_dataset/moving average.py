import pandas as pd
import numpy as np


import matplotlib.pyplot as plt

def corr(data1,data2):
    #data1 and data2 should be in np arrays#
    mean1=data1.mean()
    mean2=data2.mean()
    std1= data1.std()
    std2= data2.std()
    corr =((data1*data2).mean()-mean1*mean2)/(std1*std2)
    return corr


totaldataset_file_path = '../Dataset/total_dataset.csv'  #'20181001':'20211201'  “time,high,low,open,volumefrom,volumeto,close”
totaldataset_df = pd.read_csv(totaldataset_file_path, parse_dates=[0], index_col=0)
totaldataset_df.columns = ["high","low","open","volumefrom","volumeto","close","top_tier_volume_quote","top_tier_volume_base","top_tier_volume_total",
                    "cccagg_volume_quote","cccagg_volume_base","cccagg_volume_total",
                    "total_volume_quote","total_volume_base","total_volume_total","zero_balance_addresses_all_time","unique_addresses_all_time",
                    "new_addresses","active_addresses","transaction_count","transaction_count_all_time","large_transaction_count",
                    "average_transaction_value","block_height","hashrate",
                    "difficulty","block_time","block_size","current_supply",
                    "comments","posts","followers","points"]


#
# ss = MinMaxScaler()
# data_scaler = ss.fit_transform(totaldataset_df)   # ss.fit_transform(X)  normalize(X)
# data_scaler_df=pd.DataFrame(data_scaler)
# data_scaler_df.index=totaldataset_df.index
# data_scaler_df.columns = totaldataset_df.columns
#
#
# for a1 in data_scaler_df.columns:
#     for a2 in data_scaler_df.columns:
#         if a1 != a2 and a1=='close':
#             score,pvalue,_=coint(data_scaler_df[a1], data_scaler_df[a2])
#             correlation = corr(data_scaler_df[a1], data_scaler_df[a2])
#             print('correlation between %s and %s is %f'%(a1, a2,correlation))
#             print('Cointegration between %s and %s is %f'%(a1,a2,pvalue))
#
# corrMatrix = data_scaler_df.corr()
#
# corrMatrix.to_csv("corrMatrix.csv", index=True)
#
# print(corrMatrix)
#
# sn.heatmap(corrMatrix, annot=False)
# plt.show()
#
# print()



#-----------------------------------------------funing-----------------------------------------------------------------
# painting all
# totaldataset_df.plot(subplots=True,figsize=(30, 36))
# plt.show()

# data=totaldataset_df[['open', 'high', 'low', 'close', 'volumefrom', 'top_tier_volume_base', 'block_time', 'difficulty', 'hashrate']]
# data=data.loc['2018-10-01':'2021-12-01']
# scatter_matrix(data[['close','volumefrom','top_tier_volume_base','block_time','difficulty',"hashrate"]],figsize=(12, 12))
# plt.show()


# data=totaldataset_df[['open', 'high', 'low', 'close', 'volumefrom', 'top_tier_volume_base', 'block_time', 'difficulty', 'hashrate']]
# data=data.loc['2018-10-01':'2021-12-01']
# cov=np.corrcoef(data[['close','volumefrom','top_tier_volume_base','block_time','difficulty',"hashrate"]].T)
# print(cov)
#
# # cov=np.corrcoef(data[['close','volumefrom','top_tier_volume_base','block_time','difficulty',"hashrate"]].T)
# img=plt.matshow(cov)
# plt.colorbar(img,ticks=[-1,0,1])
# plt.show()

#
# def daily_return(data):
#     #take tail to drop head NA
#     return data.pct_change(1).dropna()
# # dt=daily_return(data)
#
#
# daily_r=daily_return(totaldataset_df[["close"]])
# daily_r.columns=['returns']
#
# totaldataset_df = pd.concat([totaldataset_df,daily_r],axis=1)
#
# totaldataset_df["z_score"]=(totaldataset_df["returns"]-totaldataset_df["returns"].mean())/totaldataset_df["returns"].std()
#
# val1=totaldataset_df[totaldataset_df["z_score"]>=3].shape[0]
# val2=totaldataset_df[totaldataset_df["z_score"]<=-3].shape[0]
# print("No. of outlier in bitcoin",val1+val2)
#-----------------------------------------------funing-----------------------------------------------------------------






# totaldataset_df['active_addresses'].plot(grid=True,color='red',label='active addresses',figsize=(20, 12))

totaldataset_df['actadd7'] = totaldataset_df['active_addresses'].rolling(window=7,min_periods=1).mean()
totaldataset_df['actadd14'] = totaldataset_df['active_addresses'].rolling(window=14,min_periods=1).mean()
totaldataset_df['actadd21'] = totaldataset_df['active_addresses'].rolling(window=21,min_periods=1).mean()
totaldataset_df['actadd28'] = totaldataset_df['active_addresses'].rolling(window=28,min_periods=1).mean()

totaldataset_df['volumefrom7']=totaldataset_df['volumefrom'].rolling(window=7,min_periods=1).mean()
totaldataset_df['volumefrom14'] = totaldataset_df['volumefrom'].rolling(window=14,min_periods=1).mean()

totaldataset_df['volumeto7']=totaldataset_df['volumeto'].rolling(window=7,min_periods=1).mean()
totaldataset_df['volumeto14']=totaldataset_df['volumeto'].rolling(window=14,min_periods=1).mean()

totaldataset_df['large_transaction_count7']=totaldataset_df['large_transaction_count'].rolling(window=7,min_periods=1).mean()


totaldataset_df['difficulty7']=totaldataset_df['difficulty'].rolling(window=7,min_periods=1).mean()
totaldataset_df['difficulty14']=totaldataset_df['difficulty'].rolling(window=14,min_periods=1).mean()
totaldataset_df['difficulty28']=totaldataset_df['difficulty'].rolling(window=28,min_periods=1).mean()

# totaldataset_df['total_volume_quote7']=totaldataset_df['total_volume_quote'].rolling(window=7,min_periods=1).mean()
# totaldataset_df['total_volume_quote14']=totaldataset_df['total_volume_quote'].rolling(window=14,min_periods=1).mean()
# totaldataset_df['total_volume_quote28']=totaldataset_df['total_volume_quote'].rolling(window=28,min_periods=1).mean()


totaldataset_df['average_transaction_value28']=totaldataset_df['average_transaction_value'].rolling(window=28,min_periods=1).mean()

totaldataset_df['closesma3']=totaldataset_df['close'].rolling(window=3,min_periods=1).mean()
totaldataset_df['closesma5']=totaldataset_df['close'].rolling(window=5,min_periods=1).mean()
totaldataset_df['closesma7']=totaldataset_df['close'].rolling(window=7,min_periods=1).mean()
totaldataset_df['closesma14']=totaldataset_df['close'].rolling(window=14,min_periods=1).mean()
totaldataset_df['closesma21']=totaldataset_df['close'].rolling(window=21,min_periods=1).mean()
totaldataset_df['closesma28']=totaldataset_df['close'].rolling(window=28,min_periods=1).mean()
totaldataset_df['closesma60']=totaldataset_df['close'].rolling(window=60,min_periods=1).mean()

#block_time
totaldataset_df['block_time7']=totaldataset_df['block_time'].rolling(window=7,min_periods=1).mean()
totaldataset_df['block_time14']=totaldataset_df['block_time'].rolling(window=14,min_periods=1).mean()
totaldataset_df['block_time28']=totaldataset_df['block_time'].rolling(window=28,min_periods=1).mean()
totaldataset_df['block_time60']=totaldataset_df['block_time'].rolling(window=60,min_periods=1).mean()
totaldataset_df['block_time100']=totaldataset_df['block_time'].rolling(window=100,min_periods=1).mean()

#block_size
totaldataset_df['block_size7']=totaldataset_df['block_size'].rolling(window=7,min_periods=1).mean()
totaldataset_df['block_size14']=totaldataset_df['block_size'].rolling(window=14,min_periods=1).mean()
totaldataset_df['block_size28']=totaldataset_df['block_size'].rolling(window=28,min_periods=1).mean()
totaldataset_df['block_size60']=totaldataset_df['block_size'].rolling(window=60,min_periods=1).mean()
totaldataset_df['block_size100']=totaldataset_df['block_size'].rolling(window=100,min_periods=1).mean()

#comments
totaldataset_df['comments3']=totaldataset_df['comments'].rolling(window=3,min_periods=1).mean()
totaldataset_df['comments5']=totaldataset_df['comments'].rolling(window=5,min_periods=1).mean()
totaldataset_df['comments7']=totaldataset_df['comments'].rolling(window=7,min_periods=1).mean()
# totaldataset_df['difficulty28']=totaldataset_df['total_volume_quote'].rolling(window=28,min_periods=1).mean()
# total_volume_quote
#Normalization------------------------------Data Processing----------------------
from sklearn.preprocessing import StandardScaler, MinMaxScaler,LabelEncoder

stander_encoder = StandardScaler()

# Close_price=stander_encoder.fit_transform(np.array(totaldataset_df['close']) .reshape(-1, 1) )

Close_price=stander_encoder.fit_transform(np.array(totaldataset_df['close']) .reshape(-1, 1) )
#active_addresses
actadd=stander_encoder.fit_transform(np.array( totaldataset_df['active_addresses']).reshape(-1, 1) )
actadd7=stander_encoder.fit_transform(np.array( totaldataset_df['actadd7']).reshape(-1, 1) )
actadd14=stander_encoder.fit_transform(np.array(totaldataset_df['actadd14']) .reshape(-1, 1) )
actadd21=stander_encoder.fit_transform(np.array(totaldataset_df['actadd21']) .reshape(-1, 1) )
actadd28=stander_encoder.fit_transform(np.array(totaldataset_df['actadd28']) .reshape(-1, 1) )
#volumefrom  volumeto
volumefrom7=stander_encoder.fit_transform(np.array(totaldataset_df['volumefrom7']) .reshape(-1, 1) )
volumeto7=stander_encoder.fit_transform(np.array(totaldataset_df['volumeto7']) .reshape(-1, 1) )

#large_transaction_count
large_transaction_count7=stander_encoder.fit_transform(np.array(totaldataset_df['large_transaction_count7']) .reshape(-1, 1) )

#difficulty
difficulty7=stander_encoder.fit_transform(np.array(totaldataset_df['difficulty7']) .reshape(-1, 1) )
difficulty14=stander_encoder.fit_transform(np.array(totaldataset_df['difficulty14']) .reshape(-1, 1) )
difficulty28=stander_encoder.fit_transform(np.array(totaldataset_df['difficulty28']) .reshape(-1, 1) )

#average_transaction_value

average_transaction_value28=stander_encoder.fit_transform(np.array(totaldataset_df['average_transaction_value28']) .reshape(-1, 1) )

closesma3=stander_encoder.fit_transform(np.array(totaldataset_df['closesma3']) .reshape(-1, 1) )
closesma5=stander_encoder.fit_transform(np.array(totaldataset_df['closesma5']) .reshape(-1, 1) )
closesma7=stander_encoder.fit_transform(np.array(totaldataset_df['closesma7']) .reshape(-1, 1) )
closesma14=stander_encoder.fit_transform(np.array(totaldataset_df['closesma14']) .reshape(-1, 1) )
closesma21=stander_encoder.fit_transform(np.array(totaldataset_df['closesma21']) .reshape(-1, 1) )
closesma28=stander_encoder.fit_transform(np.array(totaldataset_df['closesma28']) .reshape(-1, 1) )
closesma60=stander_encoder.fit_transform(np.array(totaldataset_df['closesma60']) .reshape(-1, 1) )




# #total_volume_quote  -----------------------------problem
# total_volume_quote=stander_encoder.fit_transform(np.array(totaldataset_df['total_volume_quote']) .reshape(-1, 1) )
# total_volume_quote7=stander_encoder.fit_transform(np.array(totaldataset_df['total_volume_quote7']) .reshape(-1, 1) )
# total_volume_quote14=stander_encoder.fit_transform(np.array(totaldataset_df['total_volume_quote14']) .reshape(-1, 1) )
# total_volume_quote28=stander_encoder.fit_transform(np.array(totaldataset_df['total_volume_quote28']) .reshape(-1, 1) )


#block_time
block_time7=stander_encoder.fit_transform(np.array(totaldataset_df['block_time7']) .reshape(-1, 1) )
block_time14=stander_encoder.fit_transform(np.array(totaldataset_df['block_time14']) .reshape(-1, 1) )
block_time28=stander_encoder.fit_transform(np.array(totaldataset_df['block_time28']) .reshape(-1, 1) )
block_time60=stander_encoder.fit_transform(np.array(totaldataset_df['block_time60']) .reshape(-1, 1) )
block_time100=stander_encoder.fit_transform(np.array(totaldataset_df['block_time100']) .reshape(-1, 1) )

#block_size
block_size7=stander_encoder.fit_transform(np.array(totaldataset_df['block_size7']) .reshape(-1, 1) )
block_size14=stander_encoder.fit_transform(np.array(totaldataset_df['block_size14']) .reshape(-1, 1) )
block_size28=stander_encoder.fit_transform(np.array(totaldataset_df['block_size28']) .reshape(-1, 1) )
block_size60=stander_encoder.fit_transform(np.array(totaldataset_df['block_size60']) .reshape(-1, 1) )
block_size100=stander_encoder.fit_transform(np.array(totaldataset_df['block_size100']) .reshape(-1, 1) )

#comments
comments3=stander_encoder.fit_transform(np.array(totaldataset_df['comments3']) .reshape(-1, 1) )
comments5=stander_encoder.fit_transform(np.array(totaldataset_df['comments5']) .reshape(-1, 1) )
comments7=stander_encoder.fit_transform(np.array(totaldataset_df['comments7']) .reshape(-1, 1) )



plt.figure(figsize=(30, 18))
plt.title('Testing')
# plt.plot(lstm_baseline_ytest_pred, label="LSTM_Baseline_Preds",color='Blue')
# # plt.plot(totaldataset_df['actadd7'] , label="active_addresses 7",color='Green')
# # plt.plot(totaldataset_df['actadd14'] , label="active_addresses 14",color='Red')
plt.plot(totaldataset_df['actadd21'] , label="active_addresses 21",color='Brown')
# # plt.plot(totaldataset_df['close'], label="Close Prices",color='Blue')
# # plt.legend()
# # plt.show()
#
# # plt.plot(actadd.reshape(-1, 1)  , label="active_addresses",color='Pink')
# # plt.plot(actadd7.reshape(-1, 1)  , label="active_addresses 7",color='Green')
plt.plot(actadd14.reshape(-1, 1)  , label="active_addresses 14",color='Blue')   #*Better-------------
# # plt.plot(actadd21.reshape(-1, 1)  , label="active_addresses 21",color='Brown')
# # plt.plot(actadd28.reshape(-1, 1)  , label="active_addresses 28",color='Black')
#
plt.plot(volumefrom7.reshape(-1, 1)  , label="volumefrom 7 ",color='fuchsia') #*Better-------------

plt.plot(volumefrom7.reshape(-1, 1)  , label="volumefrom 7 ",color='Green')#*Better-------------

#difficulty
# plt.plot(difficulty7.reshape(-1, 1)  , label="difficulty  7",color='Blue')#*low Better-------------
# plt.plot(difficulty14.reshape(-1, 1)  , label="difficulty 14",color='Green')
# plt.plot(difficulty28.reshape(-1, 1)  , label="difficulty 28",color='Brown')
#
plt.plot(large_transaction_count7.reshape(-1, 1)  , label="large_transaction_count 7",color='darkviolet') #*Better-------------

# plt.plot(total_volume_quote7.reshape(-1, 1)  , label="total_volume_quote",color='Blue')  -------------problem


# plt.plot(average_transaction_value7.reshape(-1, 1)  , label="average_transaction_value 7",color='Blue')
# plt.plot(average_transaction_value14.reshape(-1, 1)  , label="average_transaction_value 14",color='Green')
# plt.plot(average_transaction_value28.reshape(-1, 1)  , label="average_transaction_value 28",color='Brown')#*Better-------------

#closesma
# plt.plot(closesma3.reshape(-1, 1)  , label="closesma  3",color='fuchsia')
# plt.plot(closesma5.reshape(-1, 1)  , label="closesma  5",color='black')
# plt.plot(closesma7.reshape(-1, 1)  , label="closesma  7",color='Blue')
# plt.plot(closesma14.reshape(-1, 1)  , label="closesma 14",color='Green')
# plt.plot(closesma21.reshape(-1, 1)  , label="closesma 21",color='Brown')
# plt.plot(closesma28.reshape(-1, 1)  , label="closesma 28",color='darkviolet')
# plt.plot(closesma60.reshape(-1, 1)  , label="closesma 60",color='olive')

#block_time
# plt.plot(block_time7.reshape(-1, 1)  , label="block_time 7",color='Blue')
# plt.plot(block_time14.reshape(-1, 1)  , label="block_time 14",color='Green')
# plt.plot(block_time28.reshape(-1, 1)  , label="block_time 28",color='Brown')#*Better-------------
# plt.plot(block_time60.reshape(-1, 1)  , label="block_time 60",color='Brown')
# plt.plot(block_time100.reshape(-1, 1)  , label="block_time 100",color='Brown')

#block_size
# plt.plot(block_size7.reshape(-1, 1)  , label="block_size 7",color='Blue')
# plt.plot(block_size14.reshape(-1, 1)  , label="block_size 14",color='Green')
# plt.plot(block_size28.reshape(-1, 1)  , label="block_size 28",color='Brown')#*Better-------------
# plt.plot(block_size60.reshape(-1, 1)  , label="block_size 60",color='Brown')
# plt.plot(block_size100.reshape(-1, 1)  , label="block_size 100",color='Brown')

# plt.plot(comments3.reshape(-1, 1)  , label="comments 7",color='Blue')
# plt.plot(comments5.reshape(-1, 1)  , label="comments 5",color='Green')
# plt.plot(comments7.reshape(-1, 1)  , label="comments 7",color='Brown')


plt.plot(Close_price.reshape(-1, 1) , label="Close Prices",color='red')
plt.legend()
plt.savefig(r'C:\Users\ssel512\Desktop\save.jpg')
plt.show()





# totaldataset_df[['close','active_addresses']].plot(secondary_y='active addresses',grid=True,figsize=(30, 18))
# plt.title('2016-2017 close and active_addresses', fontsize='9')
# plt.title('20181-0-01---2021-12-01 volume', fontsize='9')
# plt.ylabel('volume', fontsize='8')
# plt.xlabel('date', fontsize='8')
# plt.legend(loc='best',fontsize='small')
# plt.show()



# plt.subplot(2,1,1)
# plt.figure(1)
# totaldataset_df[['close','active_addresses']].plot(secondary_y='active addresses',grid=True,figsize=(30, 18))
# plt.title('2016-2017 close and active_addresses', fontsize='9')
# plt.subplot(2,1,2)
# totaldataset_df[['close','transaction_count']].plot(secondary_y='transactioncount',grid=True,figsize=(30, 18))
# plt.title('2016-2017 close and transaction_count', fontsize='9')
# plt.show()


# totaldataset_df = ["high","low","open","close","top_tier_volume_quote","top_tier_volume_base","top_tier_volume_total",
#                     "new_addresses","active_addresses","transaction_count","transaction_count_all_time","large_transaction_count",
#                     "average_transaction_value","block_height","hashrate",
#                     "difficulty","block_time","block_size","current_supply",
#                     "comments","posts","followers","points"]

# plt.figure(figsize=(16,8))
# plt.rcParams.update({'font.size':10})
# plt.xticks(rotation=45)
# ax = plt.axes()
# ax.xaxis.set_major_locator(plt.MaxNLocator(20))
# plt.plot(totaldataset_df)
# plt.title('Performance of cryptocurrencies')
# plt.legend(totaldataset_df)
# plt.xlabel("Date")
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# ax.grid(True)
#
# plt.show()



# # painting K line
# def pandas_candlestick_ohlc(stock_data, otherseries=None):
#     mondays = WeekdayLocator(MONDAY)
#     alldays = DayLocator()
#     dayFormatter = DateFormatter('%d')
#
#     fig, ax = plt.subplots()
#     fig.subplots_adjust(bottom=0.2)
#     if stock_data.index[-1] - stock_data.index[0] < pd.Timedelta('730 days'):
#         weekFormatter = DateFormatter('%b %d')
#         ax.xaxis.set_major_locator(mondays)
#         ax.xaxis.set_minor_locator(alldays)
#     else:
#         weekFormatter = DateFormatter('%b %d, %Y')
#     ax.xaxis.set_major_formatter(weekFormatter)
#     ax.grid(True)
#
#
#     stock_array = np.array(stock_data.reset_index()[['date', 'open', 'high', 'low', 'close']])
#     stock_array[:, 0] = date2num(stock_array[:, 0])
#     candlestick_ohlc(ax, stock_array, colorup="red", colordown="green", width=0.6)
#
#     if otherseries is not None:
#         for each in otherseries:
#             plt.plot(stock_data[each], label=each)
#         plt.legend()
#
#     ax.xaxis_date()
#     ax.autoscale_view()
#     plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
#     plt.show()
#
#
#