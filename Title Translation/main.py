import os
origin = '/Users/jaoming/Active Projects/Shopee Challenge/Title Translation'
data_dir = '/Users/jaoming/Active Projects/Shopee Challenge/Title Translation/Product Translation dataset'

import pandas as pd                              # for importing datasets
import seaborn as sns                            # for quick visualisation
import matplotlib.pyplot as plt

os.chdir(data_dir)
def basic_clean(dataframe):
       """
       Does basic preprocessing for the data
       
       Parameters:
              dataframe : pandas df
                     the dataframe in which would be cleaned
                     
       Returns:
              output : pandas df
                     a cleaned dataframe
       """
       dataframe = dataframe.astype(str)
       dataframe = dataframe.drop_duplicates()
       dataframe = dataframe.dropna()
       dataframe.reset_index(inplace = True, drop = True)
       return dataframe
train_ch = pd.read_csv('train_tcn.csv')
train_ch = basic_clean(train_ch)
train_en = pd.read_csv('train_en.csv')
train_en = basic_clean(train_en)

# data exploration
def plot_categorycount():
       """
       Plot the category counts
       """
       fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (6, 18))               
       
       chinese = train_ch['category']
       english = train_en['category']
       
       h_chinese = sns.countplot(y = chinese, order = chinese.value_counts().index, ax = ax[0])
       for ind, label in enumerate(h_chinese.get_xticklabels()):
              if ind % 10 == 0:  # every 10th label is kept
                     label.set_visible(True)
              else:
                     label.set_visible(False)
       h_chinese.set_title('Category Counts (Chinese)')
       
       h_english = sns.countplot(y = english, order = english.value_counts().index, ax = ax[1])
       for ind, label in enumerate(h_english.get_xticklabels()):
              if ind % 20 == 0:  # every 10th label is kept
                     label.set_visible(True)
              else:
                     label.set_visible(False)
       h_english.set_title('Category Counts (English)')

def plot_sentlengths():
       """
       Plot the sentence lengths
       """
       fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (8, 18))               
       
       chinese = train_ch['product_title'].apply(len)
       english = train_en['product_title'].apply(len)
       
       h_chinese = sns.countplot(chinese, ax = ax[0])
       for ind, label in enumerate(h_chinese.get_xticklabels()):
              if ind % 10 == 0:  # every 10th label is kept
                     label.set_visible(True)
              else:
                     label.set_visible(False)
       h_chinese.set_title('Sentence Lengths (Chinese)')
       
       h_english = sns.countplot(english, ax = ax[1])
       for ind, label in enumerate(h_english.get_xticklabels()):
              if ind % 20 == 0:  # every 10th label is kept
                     label.set_visible(True)
              else:
                     label.set_visible(False)
       h_english.set_title('Sentence Lengths (English)')

plot_categorycount()
plot_sentlengths()

# example for fastText and MUSE
import fasttext                           # https://fasttext.cc/docs/en/unsupervised-tutorial.html
import numpy as np

## creating the appropriate dataset
def create_text_dataset(name, data):
       """
       To create the appropriate dataset for fastText
       
       Parameters:
              name : string
                     Name of the file that is to be created (w/o the .txt)
                     
              data : Series
                     A series of the text data that is to be written into a .txt file
       
       Returns:
              Output : None
                     Writes a `name`.txt file into the current directory
       """
       file = open(name + '.txt', 'w')
       for row in data:
              file.write(row + '\n')
       file.close()

create_text_dataset('english', train_en['product_title'])
create_text_dataset('chinese', train_ch['product_title'])

## training models
en_model = fasttext.train_unsupervised(
       'english.txt', 
       'skipgram', 
       minn = 1, 
       maxn = 8, 
       dim = 300,
       epoch = 25,
       lr = 0.01
)
en_model.save_model("english.bin")
cn_model = fasttext.train_unsupervised(
       'chinese.txt', 
       'skipgram',
       minn = 1, 
       maxn = 8, 
       dim = 300,
       epoch = 25,
       lr = 0.01
)
cn_model.save_model("chinese.bin")

## convert bin to vec
def bin_to_vec(model, vec_fname):
       # get all words from model
       words = model.get_words()

       with open(vec_fname, 'w') as file_out:
              # the first line must contain number of total words and vector dimension
              file_out.write(str(len(words)) + " " + str(model.get_dimension()) + "\n")

              # line by line, you append vectors to VEC file
              for w in words:
                     v = model.get_word_vector(w)
                     vstr = ""
                     for vi in v:
                            vstr += " " + str(vi)
                     try:
                            file_out.write(w + vstr + '\n')
                     except:
                            pass
bin_to_vec(en_model, 'english.vec')
bin_to_vec(cn_model, 'chinese.vec')

# adversial training on terminal
# https://github.com/facebookresearch/MUSE

# loading new model if possible
en_model = fasttext.load_model('english.bin')
cn_model = fasttext.load_model('chinese.bin')

## creating the necessary variable first to save time
def get_emb_and_id2word(tgt_model):
       words = tgt_model.get_words()
       return [tgt_model.get_word_vector(word) for word in words], {i: word for i, word in enumerate(words)}
tgt_emb, tgt_id2word = get_emb_and_id2word(en_model)

## prediction models
def translate_k(word, src_model, tgt_emb, tgt_id2word, k = 5):
       print('Translations for \"%s\":' % word)
       
       # creating the vector for the chosen word
       word_emb = src_model.get_word_vector(word)
       
       # finding the correlation between the chosen word vector with the vectors of the target language
       scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
       
       # getting the top k matched words
       k_best = scores.argsort()[-k:][::-1]
       for i, idx in enumerate(k_best):
              print('%.4f - %s' % (scores[idx], tgt_id2word[idx]))

translate_k('苹果', en_model, tgt_emb, tgt_id2word)

def translate(word, src_model, tgt_emb, tgt_id2word):
       word_emb = src_model.get_word_vector(word)
       scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
       best = scores.argsort()[-1:][::-1][0]
       return tgt_id2word[best]

translate('', en_model, tgt_emb, tgt_id2word)


# using google translate
from googletrans import Translator
translator = Translator()
text = translator.translate('我喜欢运动').text

# Test set
test_data = pd.read_csv('test_tcn.csv')
test_data = test_data['text']

## using google translate
translated_result = test_data.apply(lambda x: translator.translate(x).text)
translated_result = pd.DataFrame(translated_result)
translated_result.columns = ['translation_output']
translated_result.to_csv('google_translated_version.csv', index = False)

# Evaluation
from nltk.translate.bleu_score import sentence_bleu
score = sentence_bleu('reference', 'candidate')
print(score)
