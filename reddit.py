# the following is the link to gensim topic modeling example recommended by Colin
# https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#1introduction
# we need to get a client_id and client_secret as a developer, we can follow the link below for more info
# https://docs.google.com/document/d/1t_HyOwUI69MdRo46BZIzOTsAVpI7CjD100QRqiqkGWk/edit

import praw

reddit = praw.Reddit(
    client_id = "8WMKe5l_cakWbw",
    client_secret = "LaUch7Mm0BsIejJplbbv6Y6ajbzJyw",
    user_agent = "eddie's user_agent for reddit",
)

titleList = []
for submission in reddit.subreddit("learnpython").hot(limit=20):
    # print(submission.title) #uncomment to see all the titles
    titleList.append(submission.title)

# for example: we will get 20 topics when we run the above function, and we are storing them in "titleList" for gensim to consume later on
# Ask Anything Monday - Weekly Thread
# Tips on getting job as a python developer
# Advice on how to recursively edit all the filenames in my directory with a specific extension to include the folder name, please!
# Send email notification when a script generates an error
# Looking for Dancing Link/Algorithm X hints
# Blockchain Voting
# Python calculator failed need help
# Help!! I'm trying to open a text file through python
# Anyone good at using Dash?
# Constraining a Random Walk in python, how to make it work?
# Only do something when a line isn't in any line of a text file
# Are there any books or videos that describe how to organize large projects?
# Extracting JSON
# A desperate request to help me run a python script
# I'm an idiot, how do I do linear regression with sklearn?
# "reading too much input" error on number guessing game
# Scraping Data from Video Game?
# Putting data into dataframe by new line item?
# Help with a challenge
# Need help saving user input to a file


#########################################start setting up most of the packages below, you might need to install each of them one by one using pip3
# # Run in python console
import nltk; 
# nltk.download('stopwords') # if we have downloaded 'stopwords' once, we can comment out this code for the next execution

import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

#########################################finished setting up most of the packages 


# 8. Tokenize words and Clean-up text

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(titleList))

print("After cleaning up the text: ", data_words[1:2]) ## get the first list of the list of sentences
## will get the following list of the second list since the first list is just a non-changed heading: 
# [['tips', 'on', 'getting', 'job', 'as', 'python', 'developer']]


# 9. Creating Bigram and Trigram Models
# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print("trigram example: ", trigram_mod[bigram_mod[data_words[1]]])
## will get the following list:
# ['tips', 'on', 'getting', 'job', 'as', 'python', 'developer']

# 10. Remove Stopwords, Make Bigrams and Lemmatize
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# Let’s call the functions in order.
# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)

nlp = spacy.load("en_core_web_sm")
# the below part is deprecated, we have to use the above replacement instead
# nlp = spacy.load('en', disable=['parser', 'ner'])
# OSError: [E941] Can't find model 'en'. It looks like you're trying to load a model from a shortcut, 
# which is deprecated as of spaCy v3.0. To load the model, use its full name instead:

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print("After lemmatization: ", data_lemmatized[1:2])


# 11. Create the Dictionary and Corpus needed for Topic Modeling