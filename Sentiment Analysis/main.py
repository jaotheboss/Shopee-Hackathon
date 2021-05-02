import os
import numpy as np
import pandas as pd
import seaborn as sns

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from tqdm import tqdm

origin = os.getcwd()
os.chdir('/Users/jaoming/Active Projects/Shopee Challenge/Sentiment Analysis/shopee-sentiment-analysis_dataset')

train = pd.read_csv('train_embeddings.csv')
solution = pd.read_csv('solution.csv')
test = pd.read_csv('test_embeddings.csv')

"""
Data leak method
proportions = (train['rating'].value_counts()/len(train)).sort_index()

ans_1 = [1]*int(round(proportions[1]*len(test)))
ans_2 = [2]*int(round(proportions[2]*len(test)))
ans_3 = [3]*int(round(proportions[3]*len(test)))
ans_4 = [4]*int(round(proportions[4]*len(test)))
ans_5 = [5]*int(round(proportions[5]*len(test)))

pred = ans_1 + ans_2 + ans_3 + ans_4 + ans_5
test['rating'] = pred
test[['review_id', 'rating']].to_csv('submission_1.csv', index = False)
"""

# preprocessing done on Kaggle to make use of the GPU

# balancing the dataset
temp = train.loc[train['label'] == 1, :].sample(12500)
temp = temp.append(train.loc[train['label'] == 2, :].sample(12500))
temp = temp.append(train.loc[train['label'] == 3, :].sample(12500))
temp = temp.append(train.loc[train['label'] == 4, :].sample(12500))
temp = temp.append(train.loc[train['label'] == 5, :].sample(12500))
y = temp['label']
x = temp.drop('label', axis = 1)

# shrinking the dimensions
from sklearn.decomposition import PCA
pca = PCA(n_components = 50)
xpc = pca.fit_transform(x)
print(pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_))

# Preparing the model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

train_x, test_x, train_y, test_y = train_test_split(xpc, y, test_size = 0.1, random_state = 69)

model = LogisticRegressionCV(
       Cs = 20,
       fit_intercept = False,
       n_jobs = -1,
       cv = 10,
       solver = 'saga', # check the scales of the dataset (min and max)
       tol = 0.000001,
       max_iter = 200,
       random_state = 69,
)
model = MLPClassifier(
       hidden_layer_sizes = (256, 256, 128, 128, 64, 64),
       activation = 'relu',
       solver = 'adam',
       learning_rate_init = 0.001,
       max_iter = 10,
       shuffle = False
)
model = RandomForestClassifier(
       n_estimators = 200,
       max_depth = 15,
       n_jobs = -1
)
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
accuracy_score(test_y, y_pred)

r_id = test[['review_id']]
test = test.drop('review_id', axis = 1)
test = pca.fit_transform(test)
pred = model.predict(test)
r_id['rating'] = pred
r_id.to_csv('submission_3.csv', index = False)

# binary method
temp = train.loc[train['label'] == 1, :].sample(12500)
temp = temp.append(train.loc[train['label'] == 2, :].sample(14500))
temp = temp.append(train.loc[train['label'] == 3, :].sample(35500))
temp = temp.append(train.loc[train['label'] == 4, :].sample(31250))
temp = temp.append(train.loc[train['label'] == 5, :].sample(31250))
y_b = temp['label'].apply(lambda x: 0 if x <= 3 else 1)
x = temp.drop('label', axis = 1)
pca = PCA(n_components = 50)
xpc = pca.fit_transform(x)
train_x, test_x, train_y, test_y = train_test_split(xpc, y_b, test_size = 0.1, random_state = 69)

model = LogisticRegressionCV(
       Cs = 10,
       fit_intercept = True,
       n_jobs = -1,
       cv = 10,
       solver = 'saga', # check the scales of the dataset (min and max)
       tol = 0.0001,
       max_iter = 25,
       random_state = 69,
)
model = MLPClassifier(
       hidden_layer_sizes = (512, 256, 128, 64),
       activation = 'relu',
       solver = 'adam',
       learning_rate_init = 0.001,
       max_iter = 10,
       shuffle = False
)
model = RandomForestClassifier(
       n_estimators = 200,
       max_depth = None,
       n_jobs = -1
)
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
accuracy_score(test_y, y_pred)
