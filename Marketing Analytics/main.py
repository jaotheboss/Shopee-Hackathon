import os
origin = os.getcwd()
os.chdir('data')
import pandas as pd 
import numpy as np
from dataprep.eda import plot
from datetime import datetime

train = pd.read_csv('train.csv')
users = pd.read_csv('users.csv')
test = pd.read_csv('test.csv')
os.chdir(origin)

# EDA
plot(train)

# Preprocessing
from sklearn.preprocessing import OneHotEncoder

def clean_userdata(df, age_fill = 'mean'):
       """
       To clean user data
       """
       # one hot encode the domains
       ohe = OneHotEncoder()
       domains = ohe.fit_transform(df[['domain']]).toarray()
       domains = pd.DataFrame(domains, columns = ohe.get_feature_names())
       
       # fill na for age variable
       if age_fill == 'mean':
              df['age'] = df['age'].fillna(df['age'].mean())
       else:
              df['age'] = df['age'].fillna(df['age'].median())
       
       # convert all binary to -1, 0 and 1
       df['attr_1'] = df['attr_1'].replace(0, -1).fillna(0)
       df['attr_2'] = df['attr_2'].replace(0, -1).fillna(0)
       
       # putting it all together
       df.drop('domain', inplace = True, axis = 1)
       result = pd.concat([df, domains], axis = 1)
       return result

users = clean_userdata(users)

def clean(df):
       """
       To clean the Marketing Analytics data
       """
       # extract the row ids
       row_id = df['row_id']
       
       # drop unnecessary columns
       df = df.drop('row_id', axis = 1)
       
       # split grass_date into dayofweek and weekend
       def str2day(x):
              # all emails are sent at 8am
              # train['grass_date'].apply(lambda x: x.split()[1][-5:-3]).unique()
              date = x.split()[0]
              date = datetime.strptime(date, "%Y-%m-%d")
              
              # find out which day of the week
              return date.weekday(), date.month
       
       df['dayofweek'] = df['grass_date'].apply(lambda x: str2day(x)[0])
       df['month'] = df['grass_date'].apply(lambda x: str2day(x)[1])
       df['weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
       df.drop('grass_date', axis = 1, inplace = True)
       
       # flip the scale for last_open_day, last_login_day and last_checkout_day
       def str2num(x, new):
              try:
                     x = int(x)
                     return x
              except:
                     return new
       ## for last open day
       temp = list(df['last_open_day'].unique())
       temp.remove('Never open')
       temp = max([int(i) for i in temp]) + 1
       df['last_open_day'] = df['last_open_day'].apply(lambda x: str2num(x, temp))
       
       temp = list(df['last_login_day'].unique())
       temp.remove('Never login')
       temp = max([int(i) for i in temp]) + 1
       df['last_login_day'] = df['last_login_day'].apply(lambda x: str2num(x, temp))
       
       temp = list(df['last_checkout_day'].unique())
       temp.remove('Never checkout')
       temp = max([int(i) for i in temp]) + 1
       df['last_checkout_day'] = df['last_checkout_day'].apply(lambda x: str2num(x, temp))
       
       # add in user data
       df = df.merge(users, how = 'left', on = 'user_id')
       df.drop('user_id', axis = 1, inplace = True)
       
       # one hot encode country code
       ohe = OneHotEncoder()
       country_code = ohe.fit_transform(df[["country_code"]]).toarray()
       country_code = pd.DataFrame(country_code, columns = ['cc_' + str(i) for i in ohe.get_feature_names()])
       df.drop('country_code', axis = 1, inplace = True)
       df = pd.concat([df, country_code], axis = 1)
       
       # one hot encode day of week
       dayofweek = ohe.fit_transform(df[["dayofweek"]]).toarray()
       dayofweek = pd.DataFrame(dayofweek, columns = ['dow_' + str(i) for i in ohe.get_feature_names()])
       df.drop('dayofweek', axis = 1, inplace = True)
       df = pd.concat([df, dayofweek], axis = 1)
       
       # one hot encode month
       month = ohe.fit_transform(df[["month"]]).toarray()
       month = pd.DataFrame(month, columns = ['m_' + str(i) for i in ohe.get_feature_names()])
       df.drop('month', axis = 1, inplace = True)
       df = pd.concat([df, month], axis = 1)
       
       return df, row_id
              
train, train_rowid = clean(train)
test, test_rowid = clean(test)

# Model Creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# remove some label == 1's?

y = train['open_flag']
x = train.drop('open_flag', axis = 1)
for name in x.columns:
       if name in ['attr_3', 'subject_line_length', 'last_open_day', 'last_login_day', 'last_checkout_day', 
                   'open_count_last_10_days', 'open_count_last_30_days', 'open_count_last_60_days', 
                   'login_count_last_10_days', 'login_count_last_30_days', 'login_count_last_60_days', 
                   'checkout_count_last_10_days', 'checkout_count_last_30_days', 'checkout_count_last_60_days']:
              ss = StandardScaler()
              x[name] = ss.fit_transform(x[name].values.reshape(-1, 1))
#sm = SMOTE(k_neighbors = 512, sampling_strategy = 0.2, random_state = 69)
#x_res, y_res = sm.fit_resample(x, y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 69)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV

model = LogisticRegressionCV(
       Cs = 5,
       fit_intercept = True,
       cv = 10,
       solver = 'saga',
       tol = 0.00005,
       max_iter = 300,
       n_jobs = -1
)
model = RandomForestClassifier(
       n_estimators = 350,
       criterion = 'gini',
       max_depth = 12,
       n_jobs = -1
)
model.fit(x_train, y_train)

import lightgbm as lgb
# https://neptune.ai/blog/lightgbm-parameters-guide
model = lgb.LGBMClassifier(
       boosting_type = 'dart',
       num_iterations = 350,
       num_leaves = 24,
       max_depth = 8,
       learning_rate = 0.05,
       n_estimators = 128,
       feature_fraction = 0.75, # proportion of feature to feed to each tree
       bagging_fraction = 0.5, # proportion of observations to feed to each tree
       bagging_freq = 5, # perform bagging every 'k' amount of iterations
       verbose = 1,
       random_state = 69,
       early_stopping_rounds = 10 # will stop the model if it doesnt improve in the last 'k' rounds
)
model.fit(x_train, y_train)

# Model Evaluation
from sklearn.metrics import accuracy_score, matthews_corrcoef
y_pred = model.predict(x_test)
matthews_corrcoef(y_test, y_pred)
accuracy_score(y_test, y_pred)

# Test set
for name in x.columns:
       if name in ['attr_3', 'subject_line_length', 'last_open_day', 'last_login_day', 'last_checkout_day', 
                   'open_count_last_10_days', 'open_count_last_30_days', 'open_count_last_60_days', 
                   'login_count_last_10_days', 'login_count_last_30_days', 'login_count_last_60_days', 
                   'checkout_count_last_10_days', 'checkout_count_last_30_days', 'checkout_count_last_60_days']:
              ss = StandardScaler()
              test[name] = ss.fit_transform(test[name].values.reshape(-1, 1))
test['m_x0_7'] = [0]*test.shape[0]
test['m_x0_8'] = [0]*test.shape[0]

test_pred = model.predict(test)
test_result = pd.DataFrame({'row_id': test_rowid, 'open_flag': test_pred})
test_result.to_csv('submission_9 (LGBM).csv', index = False)

