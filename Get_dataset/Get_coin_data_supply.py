import requests
import pandas as pd
from datetime import datetime
import os 
import csv 



def get_all_responses(top_coins, api_url_prefix, api_url_postfix):
    '''
    Get the responses from the crypto compare api for the input list of top coins
    
    Args:
        top_coins: List
            A predefined list of top cryptocurrencies for which the historical data will be fetched using the crypto compare api
        api_url_prefix: String
            This is the srting prefix to call the cryptocompare api in which the value of id from top_coins list will be added along with the api_url_postfix
        api_url_prefix: String
            This is the srting postfix to call the cryptocompare api  which is added at the end of api_url_prefix and id from top_coins list
            
    Returns:
        final_list: List[List]
            A final List of list is returned containing the historical data of [date, coin_type,circulation_supply]
    '''
    final_list=[]
    other_list=[]
    for id in top_coins:
        response=requests.get(api_url_prefix + id + api_url_postfix)
        responses=response.json() 
        try: 
            for response in responses['Data']['Data']:
                final_list.append([response['time'], response['symbol'], response['current_supply']])
        except:
            other_list.append(id)   
        
    
    return final_list


def get_latest_responses(top_coins, api_url_prefix_latest, api_url_postfix_latest):
    '''

    Get latest data from the crypto compare api using the input list of top coins

    Args:
      top_coins: List
          A predefined list of top cryptocurrencies for which the latest (today's) data will be fetched using the crypto compare api
      api_url_prefix_latest: String
          This is the srting prefix to call the cryptocompare api in which the value of id from top_coins list will be added along with the api_url_postfix
      api_url_prefix_latest: String
          This is the srting postfix to call the cryptocompare api  which is added at the end of api_url_prefix and id from top_coins list

    Returns:
      final_list: List[List]
          A final List of list is returned containing the Latest data of [date, coin_type,circulation_supply] (should be one list for eachcoin)

    '''
    final_list = []
    for id in top_coins:
        response = requests.get(api_url_prefix_latest + id + api_url_postfix_latest)
        responses = response.json()

        # try:
        if responses['Response'] == 'Success':
            final_list.append([responses['Data']['time'],
                               responses['Data']['symbol'],
                               responses['Data']['current_supply']])
        # except:
        #     print(id, end=' ')

    return final_list

def get_OHLCV(getOHLCV):

    final_list = []
    response = requests.get(getOHLCV)
    responses = response.json()
             # #'time','high','low', 'open', 'volumefrom','volumeto','close'
    for response in responses['Data']['Data']:
        final_list.append([response['time'],
                           response['high'],
                           response['low'],
                           response['open'],
                           response['volumefrom'],
                           response['volumeto'],
                           response['close']])


    return final_list
def get_Volume(getVolume):

    final_list = []
    response = requests.get(getVolume)
    responses = response.json()
    # final_ohlcv = pd.DataFrame(final_list, columns=['time', 'top_tier_volume_quote', 'top_tier_volume_base',
    #                                                 'top_tier_volume_total',
    #                                                 'cccagg_volume_quote', 'cccagg_volume_base', 'cccagg_volume_total',
    #                                                 'total_volume_quote', 'total_volume_base', 'total_volume_total'])
    # temp= responses['Data']
    for response in responses['Data']:
        final_list.append([response['time'],
                           response['top_tier_volume_quote'],
                           response['top_tier_volume_base'],
                           response['top_tier_volume_total'],
                           response['cccagg_volume_quote'],
                           response['cccagg_volume_base'],
                           response['cccagg_volume_total'],
                           response['total_volume_quote'],
                           response['total_volume_base'],
                           response['total_volume_total']])


    return final_list

def get_Transaction(getTransaction):

    final_list = []
    response = requests.get(getTransaction)
    responses = response.json()
    # final_Volume = pd.DataFrame(final_list, columns=['zero_balance_addresses_all_time','unique_addresses_all_time','new_addresses','active_addresses',
    #                                                 'transaction_count','transaction_count_all_time','large_transaction_count',
    #                                                 'average_transaction_value','block_height','hashrate',
    #                                                 'difficulty','block_time','block_size','current_supply'])
    for response in responses['Data']['Data']:
        final_list.append([response['time'],
                           response['zero_balance_addresses_all_time'],
                           response['unique_addresses_all_time'],
                           response['new_addresses'],
                           response['active_addresses'],
                           response['transaction_count'],
                           response['transaction_count_all_time'],
                           response['large_transaction_count'],
                           response['average_transaction_value'],
                           response['block_height'],
                           response['hashrate'],
                           response['difficulty'],
                           response['block_time'],
                           response['block_size'],
                           response['current_supply']])


    return final_list

def get_Sentiment(getTransaction):

    final_list = []
    response = requests.get(getTransaction)
    responses = response.json()
    # final_Volume = pd.DataFrame(final_list, columns=['zero_balance_addresses_all_time','unique_addresses_all_time','new_addresses','active_addresses',
    #                                                 'transaction_count','transaction_count_all_time','large_transaction_count',
    #                                                 'average_transaction_value','block_height','hashrate',
    #                                                 'difficulty','block_time','block_size','current_supply'])
    temp=responses['Data']['inOutVar']['sentiment']
    # final_list.append(responses['Data']['time'])

    for response in responses['Data']:
        final_list.append([responses['Data']['time'],
                           responses['Data']['inOutVar']['sentiment'],
                           responses['Data']['inOutVar']['value'],
                           responses['Data']['largetxsVar']['sentiment'],
                           responses['Data']['largetxsVar']['value'],
                           responses['Data']['addressesNetGrowth']['sentiment'],
                           responses['Data']['addressesNetGrowth']['value'],
                           responses['Data']['concentrationVar']['sentiment'],
                           responses['Data']['concentrationVar']['value']])


    return final_list

def get_SocialStats(getSocialStats):

    final_list = []
    response = requests.get(getSocialStats)
    responses = response.json()
    for response in responses['Data']:
        final_list.append([response['time'],
                           response['comments'],
                           response['posts'],
                           response['followers'],
                           response['points']])


    return final_list

if __name__=="__main__":
    print("Code running")
    # defining list of top coins
    # top_coins=['btc' , 'eth' , 'usdt' , 'usdc', 'bnb', 'busd', 'xrp', 'ada', 'sol', 'doge', 'dot', 'shib', 'dai', 'matic', 'avax', 'trx', 'steth', 'wbtc', 'leo', 'ltc' , 'ftt', 'okb', 'cro', 'link', 'uni', 'etc', 'near', 'atom', 'xlm', 'xmr', 'algo', 'bch', 'xcn', 'tfuel', 'flow', 'vet', 'ape', 'sand', 'icp', 'mana', 'hbar', 'xtz', 'frax', 'qnt', 'fil', 'axs', 'aave', 'egld', 'tusd', 'theta']

    # getOHLCV='https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&limit=2000&tsym=USD&toTs=1638288000'   #1638288000  1538323200
    #
    # print('~~~~~~~~~getOHLCV~~~~~~~')
    # # api call to extract all historical data
    # final_list = get_OHLCV(getOHLCV)
    # print('API call made to extract OHLCV data')
    # final_ohlcv = pd.DataFrame(final_list, columns=['time','high','low', 'open', 'volumefrom','volumeto','close'])
    # # converting the date from unix format to datetime
    # final_ohlcv['time'] = pd.to_datetime(final_ohlcv['time'], unit='s')
    # # saving the historical data in a csv file
    # final_ohlcv.to_csv("final_ohlcv_data_BTC.csv", index=False)
    # print('File saved in csv')

    # getVolume='https://min-api.cryptocompare.com/data/symbol/histoday?fsym=BTC&tsym=USD&limit=2000&toTs=1638288000'
    #
    # print('~~~~~~~~~getVolume~~~~~~~')
    # # api call to extract all historical data
    # final_list = get_Volume(getVolume)
    # print('API call made to extract OHLCV data')
    # final_Volume = pd.DataFrame(final_list, columns=['time','top_tier_volume_quote','top_tier_volume_base','top_tier_volume_total',
    #                                                 'cccagg_volume_quote','cccagg_volume_base','cccagg_volume_total',
    #                                                 'total_volume_quote','total_volume_base','total_volume_total'])
    # # converting the date from unix format to datetime
    # final_Volume['time'] = pd.to_datetime(final_Volume['time'], unit='s')
    # # saving the historical data in a csv file
    # final_Volume.to_csv("final_Volume_data_BTC.csv", index=False)
    # print('File saved in csv')

    # getTransaction='https://min-api.cryptocompare.com/data/blockchain/histo/day?fsym=BTC&api_key=331502a6cb6fd8e4ec8c31994a1baf2fe19ab1f7ebf9e64820522d93acc2df3d&limit=2000&toTs=1638288000'
    #
    # print('~~~~~~~~~getVolume~~~~~~~')
    # # api call to extract all historical data
    # final_list = get_Transaction(getTransaction)
    # print('API call made to extract OHLCV data')
    # final_Transaction = pd.DataFrame(final_list, columns=['time','zero_balance_addresses_all_time','unique_addresses_all_time','new_addresses','active_addresses',
    #                                                 'transaction_count','transaction_count_all_time','large_transaction_count',
    #                                                 'average_transaction_value','block_height','hashrate',
    #                                                 'difficulty','block_time','block_size','current_supply'])
    # # converting the date from unix format to datetime
    # final_Transaction['time'] = pd.to_datetime(final_Transaction['time'], unit='s')
    # # saving the historical data in a csv file
    # final_Transaction.to_csv("final_Transaction_data_BTC.csv", index=False)
    # print('File saved in csv')

    getSentiment='https://min-api.cryptocompare.com/data/tradingsignals/intotheblock/latest?fsym=BTC&limit=2000&api_key=331502a6cb6fd8e4ec8c31994a1baf2fe19ab1f7ebf9e64820522d93acc2df3d&toTs=1638288000'

    print('~~~~~~~~~getVolume~~~~~~~')
    # api call to extract all historical data
    final_list = get_Sentiment(getSentiment)
    print('API call made to extract OHLCV data')
    final_Sentiment = pd.DataFrame(final_list, columns=['time',
                                                        'inOutVarsentiment','inOutVarvalue',
                                                        'largetxsVarsentiment', 'largetxsVarvalue',
                                                        'addressesNetGrowthsentiment', 'addressesNetGrowthvalue',
                                                        'concentrationVarsentiment', 'concentrationVarvalue',
                                                        ])
    # converting the date from unix format to datetime
    final_Sentiment['time'] = pd.to_datetime(final_Sentiment['time'], unit='s')
    # saving the historical data in a csv file
    final_Sentiment.to_csv("final_Sentiment_data_BTC.csv", index=False)
    print('File saved in csv')

    # getSocialStats ='https://min-api.cryptocompare.com/data/social/coin/histo/day?coinId=7605&aggregate=1&limit=2000&api_key=331502a6cb6fd8e4ec8c31994a1baf2fe19ab1f7ebf9e64820522d93acc2df3d&toTs=1664985600'
    #
    # print('~~~~~~~~~SocialStats~~~~~~~')
    # # api call to extract all historical data
    # final_list = get_SocialStats(getSocialStats)
    # print('API call made to extract SocialStats data')
    # final_SocialStats = pd.DataFrame(final_list, columns=['time',
    #                                                     'comments','posts',
    #                                                     'followers', 'points'
    #                                                     ])
    # # converting the date from unix format to datetime
    # final_SocialStats['time'] = pd.to_datetime(final_SocialStats['time'], unit='s')
    # # saving the historical data in a csv file
    # final_SocialStats.to_csv("final_SocialStats_data_now.csv", index=False)
    # print('File saved in csv')

    #
    # # api urls to get all the data
    # api_url_prefix = 'https://min-api.cryptocompare.com/data/blockchain/histo/day?fsym='
    # api_url_postfix='&limit=2000&api_key=331502a6cb6fd8e4ec8c31994a1baf2fe19ab1f7ebf9e64820522d93acc2df3d'
    # # 'https://min-api.cryptocompare.com/data/blockchain/histo/day?fsym=BTC&limit=2000&api_key=331502a6cb6fd8e4ec8c31994a1baf2fe19ab1f7ebf9e64820522d93acc2df3d'
    # # api to get the latest data
    # api_url_prefix_latest='https://min-api.cryptocompare.com/data/blockchain/latest?fsym='
    # api_url_postfix_latest='&api_key=331502a6cb6fd8e4ec8c31994a1baf2fe19ab1f7ebf9e64820522d93acc2df3d'
    # # 'https://min-api.cryptocompare.com/data/blockchain/latest?fsym=BTC&api_key=331502a6cb6fd8e4ec8c31994a1baf2fe19ab1f7ebf9e64820522d93acc2df3d'
    # #checking if the historical data is already present in cwd
    # files_cwd=os.listdir()
    # if('current_supply_data.csv' not in files_cwd):
    #     print('CSV file not present in cwd')
    #     # api call to extract all historical data
    #     final_list=get_all_responses(top_coins, api_url_prefix, api_url_postfix)
    #     print('API call made to extract all historical data')
    #     final_df=pd.DataFrame(final_list, columns=['date', 'coin_type', 'circulation_supply'])
    #     #converting the date from unix format to datetime
    #     final_df['date']=pd.to_datetime(final_df['date'], unit='s')
    #     #saving the historical data in a csv file
    #     final_df.to_csv("current_supply_data.csv", index=False)
    #     print('File saved in csv')
    # else:
    #     print('CSV file  present in cwd')
    #     # api call made to extract only the latest data
    #     final_list=get_latest_responses(top_coins, api_url_prefix_latest, api_url_postfix_latest)
    #     latest_df=pd.DataFrame(final_list, columns=['date', 'coin_type', 'circulation_supply'])
    #     #converting the date from unix format to datetime
    #     latest_df['date']=pd.to_datetime(latest_df['date'], unit='s').dt.date
    #     # converting df again to list so that it is easier to append to the existing csv
    #     final_list=latest_df.values.tolist()
    #     # reading the existing data from the historical data csv file (current_supply_data.csv)
    #     with open('current_supply_data.csv','r') as file_object:
    #       existing_lines  = [line for line in csv.reader(file_object, delimiter=',')]
    #       file_object.close()
    #     #writing the latest data in the existing historical data csv file (current_supply_data.csv)
    #     with open('current_supply_data.csv','a') as file_object:
    #       writer_object = csv.writer(file_object)
    #       for value in final_list:
    #           #checking if the latest data is already present in the current_supply_data.csv
    #           if value not in existing_lines:
    #               writer_object.writerow(value)
    #
    #     file_object.close()
    #     print("Latest data appended to the already saved csv file")

        