import os
from dataprep.eda import plot
import pandas as pd
import time
from datetime import datetime, timedelta
import numpy as np

origin_dir = os.getcwd()
data_dir = '/Users/jaoming/Active Projects/Shopee Challenge/Logistics Analytics/logistics-shopee-code-league'

os.chdir(data_dir)
data_do = pd.read_csv('delivery_orders_march.csv')

# data exploration
plot(data_do)

# data preprocessing
## checking for nan datapoints
len(data_do['orderid'].unique()) == data_do.shape[0] 
sum(~pd.isna(data_do['pick'])) == data_do.shape[0]
sum(~pd.isna(data_do['1st_deliver_attempt'])) == data_do.shape[0]

## establishing sla mat
sla_mat = {
       'metro manila': {
              'metro manila': 3,
              'luzon': 5,
              'visayas': 7,
              'mindanao': 7
       },
       
       'luzon': {
              'metro manila': 5,
              'luzon': 5,
              'visayas': 7,
              'mindanao': 7
       },
       
       'visayas': {
              'metro manila': 7,
              'luzon': 7,
              'visayas': 7,
              'mindanao': 7
       },
       
       'mindanao': {
              'metro manila': 7,
              'luzon': 7,
              'visayas': 7,
              'mindanao': 7
       }
}

## public holidays
ph = [
       '2020-03-08',
       '2020-03-25',
       '2020-03-30',
       '2020-03-31'
]

## cleaning of data
def cleaning(dataset):
       """
       For cleaning the delivery order dataset
       """
       dataset = dataset.sort_values('pick')
       # setting the dates
       def convert_dt(epoch):
              if pd.isna(epoch):
                     return 'na'
              else:
                     temp = time.strftime('%Y-%m-%d', time.localtime(epoch))
                     return temp
       dataset['pick'] = dataset['pick'].apply(convert_dt)
       dataset['1st_deliver_attempt'] = dataset['1st_deliver_attempt'].apply(convert_dt)
       dataset['2nd_deliver_attempt'] = dataset['2nd_deliver_attempt'].apply(convert_dt)
       
       # settling the locations
       def extract_region(address):
              result = []
              address = address.lower()
              for key in sla_mat:
                     if key in address:
                            result.append(key)
              if len(result) == 1:
                     return result[0]
              else:
                     # some entries have more than one region in the address. for this case we took the end
                     result.sort(key = lambda x: address.find(x), reverse = True)
                     return result[0]    
       dataset['buyeraddress'] = dataset['buyeraddress'].apply(extract_region)
       dataset['selleraddress'] = dataset['selleraddress'].apply(extract_region)
       dataset.columns = ['orderid', 'start', '1st', '2nd', 'buyeraddress', 'selleraddress']
       dataset.reset_index(drop = True, inplace = True)
       return dataset

clean_data = cleaning(data_do) 

# main algo
def check_late(row):
       first_delta = np.busday_count(row['start'], row['1st'], weekmask = '1111110', holidays = ph)
       if first_delta > sla_mat[row['selleraddress']][row['buyeraddress']]:
              # if the first attempt was late, consider it late
              return 1
       else:
              if row['2nd'] == 'na':
                     # if the first attempt wasn't late and there's no second attempt, it's on time
                     return 0
              else:
                     # if the first attempt wasn't late but the second attempt was late, consider it late
                     second_delta = np.busday_count(row['1st'], row['2nd'], weekmask = '1111110', holidays = ph)
                     if second_delta > 3:
                            return 1
                     else:
                            return 0
clean_data['is_late'] = clean_data.apply(check_late, axis = 1)

result = clean_data.loc[:, ['orderid', 'is_late']]
result.to_csv('trial_6.csv', index = False)