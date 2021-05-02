import os
import numpy as np
import pandas as pd
import seaborn as sns

origin = os.getcwd()
os.chdir('/Users/jaoming/Active Projects/Shopee Challenge/Sentiment Analysis/shopee-sentiment-analysis_dataset')
train = pd.read_csv('train.csv')
solution = pd.read_csv('solution.csv')

review_lengths = [len(i) for i in train['review']]
def plot():
       plot = sns.countplot(review_lengths)
       for ind, tick in enumerate(plot.get_xticklabels()):
              if ind % 30 == 0:
                     tick.set_visible(True)
              else:
                     tick.set_visible(False)
plot()

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models

# preprocessing the reviews
def _PREPROCESS(text):
       """
       Function:     Acts as a sub function to the bigger _PREPROCESS.
                     This function seeks to only preprocess the questions
                     
       Input:        Questions column
       
       Returns:      A column of reprocessed questions
       """
       ## for manipulating the questions
       stop_words = stopwords.words('english')
       # stop_words.extend(['hi', 'hello', 'amp'])

       #ps = PorterStemmer()
       wnl = WordNetLemmatizer()

       contractions = {
              "ain't": "am not / are not",
              "aren't": "are not / am not",
              "can't": "cannot",
              "can't've": "cannot have",
              "'cause": "because",
              "could've": "could have",
              "couldn't": "could not",
              "couldn't've": "could not have",
              "didn't": "did not",
              "doesn't": "does not",
              "don't": "do not",
              "hadn't": "had not",
              "hadn't've": "had not have",
              "hasn't": "has not",
              "haven't": "have not",
              "he'd": "he had / he would",
              "he'd've": "he would have",
              "he'll": "he shall / he will",
              "he'll've": "he shall have / he will have",
              "he's": "he has / he is",
              "how'd": "how did",
              "how'd'y": "how do you",
              "how'll": "how will",
              "how's": "how has / how is",
              "i'd": "I had / I would",
              "i'd've": "I would have",
              "i'll": "I shall / I will",
              "i'll've": "I shall have / I will have",
              "i'm": "I am",
              "i've": "I have",
              "isn't": "is not",
              "it'd": "it had / it would",
              "it'd've": "it would have",
              "it'll": "it shall / it will",
              "it'll've": "it shall have / it will have",
              "it's": "it has / it is",
              "let's": "let us",
              "ma'am": "madam",
              "mayn't": "may not",
              "might've": "might have",
              "mightn't": "might not",
              "mightn't've": "might not have",
              "must've": "must have",
              "mustn't": "must not",
              "mustn't've": "must not have",
              "needn't": "need not",
              "needn't've": "need not have",
              "o'clock": "of the clock",
              "oughtn't": "ought not",
              "oughtn't've": "ought not have",
              "shan't": "shall not",
              "sha'n't": "shall not",
              "shan't've": "shall not have",
              "she'd": "she had / she would",
              "she'd've": "she would have",
              "she'll": "she shall / she will",
              "she'll've": "she shall have / she will have",
              "she's": "she has / she is",
              "should've": "should have",
              "shouldn't": "should not",
              "shouldn't've": "should not have",
              "so've": "so have",
              "so's": "so as / so is",
              "that'd": "that would / that had",
              "that'd've": "that would have",
              "that's": "that has / that is",
              "there'd": "there had / there would",
              "there'd've": "there would have",
              "there's": "there has / there is",
              "they'd": "they had / they would",
              "they'd've": "they would have",
              "they'll": "they shall / they will",
              "they'll've": "they shall have / they will have",
              "they're": "they are",
              "they've": "they have",
              "to've": "to have",
              "wasn't": "was not",
              "we'd": "we had / we would",
              "we'd've": "we would have",
              "we'll": "we will",
              "we'll've": "we will have",
              "we're": "we are",
              "we've": "we have",
              "weren't": "were not",
              "what'll": "what shall / what will",
              "what'll've": "what shall have / what will have",
              "what're": "what are",
              "what's": "what has / what is",
              "what've": "what have",
              "when's": "when has / when is",
              "when've": "when have",
              "where'd": "where did",
              "where's": "where has / where is",
              "where've": "where have",
              "who'll": "who shall / who will",
              "who'll've": "who shall have / who will have",
              "who's": "who has / who is",
              "who've": "who have",
              "why's": "why has / why is",
              "why've": "why have",
              "will've": "will have",
              "won't": "will not",
              "won't've": "will not have",
              "would've": "would have",
              "wouldn't": "would not",
              "wouldn't've": "would not have",
              "y'all": "you all",
              "y'all'd": "you all would",
              "y'all'd've": "you all would have",
              "y'all're": "you all are",
              "y'all've": "you all have",
              "you'd": "you had / you would",
              "you'd've": "you would have",
              "you'll": "you shall / you will",
              "you'll've": "you shall have / you will have",
              "you're": "you are",
              "you've": "you have"}

       def contract(text):
              for word in text.split():
                     if word.lower() in contractions:
                            text = text.replace(word, contractions[word.lower()])
              return text
       
       def preprocess(text_column):
              """
              Function:     This NLP pre processing function takes in a sentence,
                            replaces all the useless letters and symbols, and takes 
                            out all the stop words. This would hopefully leave only 
                            the important key words
                            
              Input:        A list of sentences
              
              Returns:      A list of sentences that has been cleaned
              """
              # Remove link,user and special characters
              # And Lemmatize the words
              new_review = []
              for review in text_column:
                     # this text is a list of tokens for the review
                     text = re.sub("@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+", ' ', str(review).lower()).strip()
                     text = contract(text).split(' ')
                     
                     # Stemming and removing stopwords
                     text = [wnl.lemmatize(i) for i in text if i not in stop_words]
                     
                     new_review.append(' '.join(text))
              return new_review
       
       text = preprocess(text)
       return text

train.drop('review_id', axis = 1, inplace = True)
train = train.sample(frac = 1).reset_index(drop = True)

y_train = train['rating']
y_train = y_train - 1

x_train = train['review']
train['review'] = _PREPROCESS(train['review'])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)
x_train = pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen = 256, padding = 'post')

# creating the model
model = models.Sequential()
model.add(layers.Embedding(
       input_dim = len(tokenizer.word_index) + 1,       # vocab size
       output_dim = 128,                                # length of embedding vector
       input_length = 256                               # length of sentence vector (after tokenization)
))
model.add(layers.Conv1D(
       filters = 64,
       kernel_size = 3,
       activation = 'relu'
))
model.add(layers.MaxPooling1D(
       pool_size = 2
))
model.add(layers.Bidirectional(
       layers.LSTM(
              units = 32,
              activation = 'tanh',
              dropout = 0,
              recurrent_dropout = 0
       )
))
model.add(layers.Dense(
       units = 5,
       activation = 'softmax'
))
model.summary()

model.compile(
       loss = 'sparse_categorical_crossentropy', 
       optimizer = 'nadam', 
       metrics = ['acc']
)

model.fit(
       x_train,
       y_train,
       batch_size = 16,
       epochs = 1,
       validation_split = 0.1
)

# test set
answers = pd.read_csv('submission_1.csv')
test = pd.read_csv('test.csv')
test_answer = pd.merge(test, answers, how = 'left', on = 'review_id').drop('review_id', axis = 1)
test_answer = test_answer.sample(frac = 1).reset_index(drop = True)

test_x = test_answer['review'][:100].apply(lambda x: eb.embed_sentence(x, max_len = 128))
test_x = pd.DataFrame(test_x.values.tolist())
test_y = test_answer['rating'][:100]

y_pred = model.predict(test_x)
accuracy_score(test_y, y_pred)


