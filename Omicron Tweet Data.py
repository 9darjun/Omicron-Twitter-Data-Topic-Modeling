%%time
# !pip install gensim==3.8.3 or higher
# !pip install pyLDAvis==3.3.1 or higher

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from IPython.display import HTML, display
import tabulate
import pandas as pd
import numpy as np
from PIL import Image
from wordcloud import WordCloud
import seaborn as sns
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis

# from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer, PorterStemmer
# from nltk.stem.porter import *

import gensim
from gensim.models import Phrases
#Prepare objects for LDA gensim implementation
from gensim import corpora
#Running LDA
from gensim import models

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import re
%matplotlib inline

data = pd.read_csv('*/omicron_tweets.csv', engine='python')
# dropping empty rows
data = data.dropna(subset=['text'])
# dropping duplicates
data = data.drop_duplicates()
print('Rows: {}, columns: {}'.format(data.shape[0], data.shape[1]))
data.head(2)

data['processed'] = ''
nltk.download('wordnet')
nltk.download('stopwords')

stop_words = stopwords.words('english')

stop_words.extend(['https'])
# initalizing the werdnet lemmatizer
lm = WordNetLemmatizer()

def processing(content):

    content = content.replace('\n', ' ').split(' ')
#  removing these punctuations from tokens like it will convert the word mode? into mode
    rx = re.compile('([&#.:?!-()])*')
    content = [rx.sub('', word) for word in content]
    
#  removing stopwords
    content = [word.strip().lower() for word in content if word.strip().lower() not in stop_words]
  #  remove words whose length is greater than 1 and or alphabetics only 
    content = [word for word in content if len(word)>1 and word.isalpha()]
    # lemmatizing the words to their basic form
    content = [lm.lemmatize(word) for word in content]

    return ' '.join(content)

for k in range(len(data)):
  data.iloc[k,-1] = processing(data.iloc[k,2])
# processed data
data.head()

b = data['processed'].tolist()
b = ' '.join(map(str, b))
# b = b.replace(',', ' ').lower().replace('-', '')

wordcloud = WordCloud(max_font_size=40, max_words=1000, background_color="white", random_state=100,
                      prefer_horizontal=0.60).generate(b.lower())
# max_font_size=40, max_words=1000, background_color="white",
                      # random_state=100, prefer_horizontal=0.50
plt.figure(figsize=(12,8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
# plt.show()
plt.savefig('word_cloud.jpg')

lengths = [len(sentence.split(' ')) if len(sentence)>0 else 0 for sentence in data['processed']]
data['Lengths'] = lengths
plt.figure(figsize=(10,6))
data['Lengths'].hist(bins=30)
sns.despine(top=True, right=True, left=False, bottom=False)
plt.title('Number of words in each tweet')
plt.xlabel('Words')
plt.ylabel('Length of tweets')
plt.savefig('length_tweets.jpg')
plt.show()

import gensim.corpora as corpora

#decomposing sentences into tokens
tokens = [sentence.split(' ') for sentence in data['processed'] ]
# training a bi gram model in order to include those bigrams as tokens who occured at least 6 times
# in the whole dataset
bigram = gensim.models.Phrases(tokens, min_count=2, threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)

# including bigrams as tokens 
sents = [ bigram_mod[token] for token in tokens]

# Create Dictionary to keep track of vocab
dct = corpora.Dictionary(tokens)

print('Unique words before filtering/after pre-processing', len(dct))
# no_below= 30
# filter the words that occure in less than 3 documents and in more the 60% of documents
dct.filter_extremes(no_below= 3, no_above=0.60 )
print('Unique words after filtering', len(dct))

# Create Corpus
corpus = [dct.doc2bow(sent) for sent in sents]

tfidf = gensim.models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

from gensim.models import CoherenceModel
import time
import os

scores = []
for k in range(3,15):
    # LDA model
    lda_model = gensim.models.LdaModel(  corpus=corpus_tfidf, num_topics=k,
                                                 id2word=dct, random_state=12)
    # to calculate score for coherence
    coherence_model_lda = CoherenceModel(model=lda_model, texts=sents, dictionary=dct, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print(k, coherence_lda)
    scores.append(coherence_lda)

selected_topics = np.argmax(scores)+3

plt.figure(figsize=(10, 5))
plt.plot(list(range(3,15)), scores, marker='o', color='green')
sns.despine(top=True, right=True, left=False, bottom=False)

plt.locator_params(integer=True)
plt.title('Coherence score vs the number of topics for LDA')
plt.xlabel('Number of topics')
plt.ylabel('Coherence Scores')
plt.savefig('lda_scores.jpg')
plt.show()

import pyLDAvis.gensim_models

lda_model = gensim.models.LdaModel(corpus=corpus_tfidf, id2word=dct, num_topics=selected_topics, 
                                           random_state=12, chunksize=128, passes=10 )

pyLDAvis.enable_notebook()
results = pyLDAvis.gensim_models.prepare(lda_model, corpus_tfidf, dct, sort_topics=False)
pyLDAvis.save_html(results, 'ldavis_english' +'.html')
results

# top words in each topic
lda_model.print_topics()

from gensim.models.nmf import Nmf

scores_nmf = []
for k in range(3,15):
    # lda mallet model
    nmf_model = Nmf(corpus_tfidf, num_topics=k, \
                                  id2word=dct, \
                                  passes=10)
    # to calculate score for coherence
    coherence_model_lda = CoherenceModel(model=nmf_model, texts=sents, dictionary=dct, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print(k, coherence_lda)
    scores_nmf.append(coherence_lda)

plt.figure(figsize=(10, 5))
plt.plot(list(range(3,15)), scores_nmf, marker='o', color='blue')
sns.despine(top=True, right=True, left=False, bottom=False)

plt.locator_params(integer=True)
plt.title('Coherence score vs the number of topics for NMF')
plt.xlabel('Number of topics')
plt.ylabel('Coherence Scores')
plt.savefig('nmf_scores.jpg')
plt.show()

selected_topics_nmf=np.argmax(scores_nmf)+3
nmf_model = Nmf(corpus=corpus_tfidf, id2word=dct, num_topics=selected_topics_nmf, 
                                           random_state=12, chunksize=128, passes=10 )

nmf_model.print_topics()

from gensim.models.lsimodel import LsiModel

scores_lsi = []
for k in range(3,15):
    # LSI model
    lsi_model = LsiModel(  corpus=corpus_tfidf, num_topics=k,power_iters=250,
                                                 id2word=dct)
    # to calculate score for coherence
    coherence_model_lsi = CoherenceModel(model=lsi_model, texts=sents, dictionary=dct, coherence='c_v')
    coherence_lsi = coherence_model_lsi.get_coherence()
    print(k, coherence_lsi)
    scores_lsi.append(coherence_lsi)

plt.figure(figsize=(10, 5))
plt.plot(list(range(3,15)), scores_lsi, marker='o', color='black')
sns.despine(top=True, right=True, left=False, bottom=False)

plt.locator_params(integer=True)
plt.title('Coherence score vs the number of topics for LSA')
plt.xlabel('Number of topics')
plt.ylabel('Coherence Scores')
plt.savefig('lsa_scores.jpg')
plt.show()

selected_topics_lsi = np.nanargmax(scores_lsi)+3
lsi_model = LsiModel(  corpus=corpus_tfidf, num_topics=selected_topics_lsi,
                                                 id2word=dct)
lsi_model.print_topics()

# judged manually from pyldavis
topics_name = ['Topic 1', 'Topic 2', 'Topic 3']

# getting the most dominant topics from a trained model
predicted_topics = lda_model[corpus_tfidf]

probs, topics = [], []
for k in predicted_topics:
  # sorting the probabilites
  k.sort(key=lambda x:x[1])
  # selecting the topic with greates probability
  topics.append(topics_name[ k[0][0] ] )

data['Topics'] = topics
data.head(2)

# 1. Wordcloud of Top N words in each topic

from matplotlib import pyplot as plt
from wordcloud import WordCloud

cloud = WordCloud(background_color='white', width=2500, height=2000,
                  max_words=28, colormap='tab10', prefer_horizontal=1.0)
  
fig, axes = plt.subplots(1, 3, figsize=(15,8), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
  
  fig.add_subplot(ax)
  if i>len(topics_name)-1:
    continue
  curr = data[data['Topics']==topics_name[i]]
  print(curr.shape)
  tokens = [tok for d in curr['processed'] for tok in d.split(' ')]
  cloud.generate(' '.join( tokens ))
      
  plt.gca().imshow(cloud)
  plt.gca().set_title( topics_name[i]+'\n')
  plt.gca().axis('off')

plt.axis('off')
plt.tight_layout()
plt.show()

def subcategory_plot(df, col):
    
    plt.figure(figsize=(10,6))
    bar_pub = list( df[col].value_counts().index[:5] )
    temp2 = df[df[col].isin(bar_pub)]

    sns.countplot(x=col, hue='Topics', data=temp2, palette="RdYlGn")
    sns.despine()

    plt.savefig('bars_{}.png'.format(col))
    plt.show()

subcategory_plot(data, 'source')

# users have been categorized according to their number of followers
data['User type']=pd.cut(data.user_followers, [1,200, 1000, 2000, 10000, 100000000], 
                             labels=['Naive', 'Average', 'Popular', 'Micro-Influencer', 'Influencer'])

subcategory_plot(data, 'User type')
