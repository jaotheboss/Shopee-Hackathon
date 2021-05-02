"""
This logic proved to be able to get 1.00 but the current implementation is really slow. 

Running the algorithm for 10 shops alone took 32.5 minutes to run.
Reasons for this could be:
1. Finding the end_index
"""

import os
os.chdir('/Users/jaoming/Active Projects/Shopee Challenge/Data Brushing')

# importing the relevant modules
import pandas as pd

# importing the data
data = pd.read_csv('order_brush_order.csv')
for col_name in data.columns:                                                                     # checking if there're any empty data points
       print(col_name)
       print(data[col_name].isna().sum())
       print()
all_shops = list(data['shopid'].unique())                                                         # all shops 

# preprocessing of the data
data['event_time'] = pd.to_datetime(data['event_time'])                                           # converting string into date object
grouped_by_shop = {}
for i in range(len(data.index)):                                                                  # gathering the transaction details (user and timing) for each order by the shop
       shop_id, user_id, transact_time = data.iat[i, 1], data.iat[i, 2], data.iat[i, 3]
       if shop_id not in grouped_by_shop.keys():
              grouped_by_shop[shop_id] = []
       grouped_by_shop[shop_id].append([transact_time, user_id])
 
result_df = {
       'shopid': [],
       'userid': []
}
for shop_id in list(grouped_by_shop.keys())[:10]:
       # gathering the transactions from this particular shopid
       shop_transactions = grouped_by_shop[shop_id]
       shop_transactions.sort()
       limit = len(shop_transactions)

       possible_culprits = []                                                                     # for the collation of users that appeared within the order brushing regions

       # finding the order transaction indexes that occur within 1 hour windows
       end_index = 0
       for start_index in range(limit - 2):                                                       # -2 because there's no point checking only the last 2 transactions. even if it's 1 user and 2 sales, the max conc rate will be 2
              start_time = shop_transactions[start_index][0]
              end_index = len([i for i in shop_transactions if i[0] <= (start_time + pd.to_timedelta(1, unit = 'h'))])
              if end_index - start_index <= 2:
                     continue                                                                     # to speed up the algo
              transactions_in_window = shop_transactions[start_index:end_index]
              n_orders = len(transactions_in_window)
              users = pd.Series([i[1] for i in transactions_in_window]).unique()
              n_unique_users = len(users)
              conc_rate = n_orders/n_unique_users
              if conc_rate >= 3:
                     for user in users:
                            if user not in possible_culprits:
                                   possible_culprits.append(user)
       
       # at this point, all the windows would have been screened and all possible order brushing users should be in possible_culprits
       result_df['shopid'].append(shop_id)
       if possible_culprits == []:
              result_df['userid'].append('0')
       else:
              shop_transactions_usersonly = pd.Series([i[1] for i in shop_transactions])
              shop_transactions_usersonly = shop_transactions_usersonly.value_counts()
              max_orders = shop_transactions_usersonly.max()
              top_culprits = list(shop_transactions_usersonly[shop_transactions_usersonly == max_orders].index)
              top_culprits = [str(i) for i in top_culprits if i in possible_culprits]
              if top_culprits == []:
                     result_df['userid'].append('0')
              else:
                     result_df['userid'].append('&'.join(top_culprits))

result_df = pd.DataFrame(result_df)







