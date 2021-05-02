import os
os.chdir('/Users/jaoming/Active Projects/Shopee Challenge/Data Brushing')

# importing the relevant packages
import pandas as pd

data = pd.read_csv('order_brush_order.csv')
for col_name in data.columns:                                                                     # checking if there're any empty data points
       print(col_name)
       print(data[col_name].isna().sum())
       print()
all_shops = list(data['shopid'].unique())                                                         # all shops 

data['event_time'] = pd.to_datetime(data['event_time'])                                           # converting string into date object
print('Data only shows these days:', data['event_time'].map(lambda x: x.date()).unique())
data = data.sort_values(by = 'event_time')                                                        # sort the data by date
data.reset_index(inplace = True, drop = True)

# counting the average difference in time between transactions
diff_times = data['event_time'].shift(-1) - data['event_time']
diff_times.dropna(inplace = True)
average_diff = diff_times.mean()
milliseconds = int(1000*average_diff.microseconds / 10**6)/1000
seconds = average_diff.seconds
print('Transactions happens every', seconds + milliseconds, 'seconds on average')




######------------------------START---------------------------------------------------------------------
# EXTRACTING THE SUSPECTED STORES BY CREATING A WINDOW THAT SLIDES THROUGH 1 HOUR INTERVALS TO FILTER OUT THOSE THAT EXCEED THE CONC RATE
# how many hours we have to look through
no_hours = int(((data['event_time'][len(data) - 1] - data['event_time'][0]).days * 24) + ((((data['event_time'][222749] - data['event_time'][0]).seconds / 60)//60) + 1))

# initialise the window
start_window = data['event_time'][0]
end_window = start_window + pd.to_timedelta(1, unit = 'h')            # window is 1 hour apart
sliding_rate = 0.005                                                 # slides the 1 hour window at 0.0025 hours intervals
suspected_stores = []

for i in range(no_hours*200):  # if we're afraid that some data might be lost 
       filtered_data = data.loc[data['event_time'] >= start_window, :]
       filtered_data = filtered_data.loc[filtered_data['event_time'] <= end_window, :]                   # getting data from that window only
       if len(filtered_data) == 0:
              continue
       else:
              shop_uniquebuyers = filtered_data.groupby('shopid').userid.nunique()                       # getting unique buyers
              shop_uniquebuyers = shop_uniquebuyers.reset_index()
              shop_uniquebuyers.columns = ['shopid', 'unique_buyers']

              shop_ordercount = filtered_data.groupby('shopid').orderid.count()                          # getting the number of orders
              shop_ordercount = shop_ordercount.reset_index()
              shop_ordercount.columns = ['shopid', 'order_count']

              conc_rate = shop_ordercount['order_count']/shop_uniquebuyers['unique_buyers']              # vector calculation to find the conc rate

              merged_data = pd.merge(shop_ordercount, shop_uniquebuyers, on = 'shopid')                  
              stores = merged_data.loc[conc_rate >= 3]['shopid']                                         # getting the shopid's that surpass the threshold
              if len(stores) != 0:
                     for i in stores:
                            if i not in suspected_stores:
                                   suspected_stores.append(i)                                            # compile all the stores together in a list
       # sliding the window by one hour
       start_window, end_window = start_window + pd.to_timedelta(sliding_rate, unit = 'h'), end_window + pd.to_timedelta(sliding_rate, unit = 'h')


# EXTRACTING OUT THE SUSPICIOUS USERS BASED ON THE SUSPECTED STORES
result_df = {
       'shopid': [],
       'userid': []
}
for store_id in suspected_stores:
       # grouping by users and counting the number of orders each user made
       user_ordercount = data.loc[data['shopid'] == store_id, ['orderid', 'userid']].groupby('userid').count().sort_values(by = 'orderid', ascending = False)
       user_ordercount = user_ordercount.reset_index()
       user_ordercount.columns = ['userid', 'order_count']

       max_order = user_ordercount.max()['order_count']        # getting the maximum order count from this shop
       suspected_users = list(user_ordercount.loc[user_ordercount['order_count'] == max_order, 'userid'])              # getting only the users that have that order count
       suspected_users.sort()                                  # sort by smallest userid to biggest 

       # appending to the result_df
       result_df['shopid'].append(store_id)
       if len(suspected_users) == 1:
              result_df['userid'].append(str(suspected_users[0]))
       else:
              str_users = list(map(lambda x: str(x), suspected_users))
              str_users = '&'.join(str_users)
              result_df['userid'].append(str_users)

# ADDING THE REST OF THE NON-SUSPECTED STORES
for i in all_shops:
       if i not in result_df['shopid']:
              result_df['shopid'].append(i)
              result_df['userid'].append('0')

result_df = pd.DataFrame(result_df)
result_df.to_csv('trial10.csv', index = False)