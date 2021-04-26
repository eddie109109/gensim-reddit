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

# Now we can scrape reddit by using the for loop below
# for example: we will get 20 topics when we run the above function EVERYDAY, and we are storing them in "titleList" for gensim to consume later on
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

titleList = []
numOfTopic = 20
# the below for loop is for scrape the data from reddit
for submission in reddit.subreddit("Psychedelics").hot(limit=numOfTopic):
    print(submission.title) #uncomment to see all the titles
    titleList.append(submission.title)

# since i do not want my data to change every day, i have decided to use the follwing 20 topics as the list for this test run, 
# we can uncomment the above function to scrape new data
# titleList = ["Ask Anything Monday - Weekly Thread",
# "Tips on getting job as a python developer",
# "Advice on how to recursively edit all the filenames in my directory with a specific extension to include the folder name, please!",
# "Send email notification when a script generates an error",
# "Looking for Dancing Link/Algorithm X hints",
# "Blockchain Voting",
# "Python calculator failed need help",
# "Help!! I'm trying to open a text file through python",
# "Anyone good at using Dash?",
# "Constraining a Random Walk in python, how to make it work?",
# "Only do something when a line isn't in any line of a text file",
# "Are there any books or videos that describe how to organize large projects?",
# "Extracting JSON",
# "A desperate request to help me run a python script",
# "I'm an idiot, how do I do linear regression with sklearn?",
# '"reading too much input" error on number guessing game',
# "Scraping Data from Video Game?",
# "Putting data into dataframe by new line item?",
# "Help with a challenge",
# "Need help saving user input to a file"]


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

print("After cleaning up the text (separate each topic into individual word and put it into a list): ")
print(data_words) 
print("**************After cleaning up the text **************")
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
print("trigram example: ")
for i in range(numOfTopic):
    print(trigram_mod[bigram_mod[data_words[i]]])
print("************** End of trigram example **************")

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

print("The result of lemmatization: ")
print(data_lemmatized[0:])

print("************** After lemmatization **************")


# 11. Create the Dictionary and Corpus needed for Topic Modeling
# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]


# View
print("View the word frequency: (corpus)")
print(corpus)
print("************** The end of the word frequency **************")
# Generate the following frequency list:
# Based on [['tip', 'get', 'job', 'python', 'developer']]
# [[(4, 1), (5, 1), (6, 1), (7, 1), (8, 1)]]


# Gensim creates a unique id for each word in the document. The produced corpus shown above is a mapping of (word_id, word_frequency).
# For example, (0, 1) above implies, word id 0 occurs once in the first document. Likewise, word id 1 occurs twice and so on.
# This is used as the input by the LDA model.

# Human readable format of corpus (term-frequency)
termFrequencyList = [[(id2word[id], freq) for id, freq in cp] for cp in corpus[0:]]
print("**************** printing hunman readable format of corpus ****************")
for i in range(len(termFrequencyList)):
    print(termFrequencyList[i])
print("**************** end of printing hunman readable format of corpus ****************")


# 12. Building the Topic Model

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)



# 13. View the topics in LDA model

# Print the Keyword in the 20 topics
print("************** Print the keyword in the 20 topics below **************")
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]  

# below is the output
# [(0,
#   '0.013*"describe" + 0.013*"book" + 0.013*"desperate" + 0.013*"json" + '
#   '0.013*"extract" + 0.013*"video" + 0.013*"project" + 0.013*"organize" + '
#   '0.013*"large" + 0.013*"run"'),
#  (1,
#   '0.070*"file" + 0.070*"help" + 0.070*"input" + 0.070*"need" + 0.070*"user" + '
#   '0.070*"save" + 0.070*"algorithm" + 0.070*"hint" + 0.070*"look" + '
#   '0.070*"link"'),
#  (2,
#   '0.059*"notification" + 0.059*"email" + 0.059*"error" + 0.059*"send" + '
#   '0.059*"script" + 0.059*"generate" + 0.059*"ask" + 0.059*"thread" + '
#   '0.059*"monday" + 0.059*"weekly"'),
#  (3,
#   '0.212*"challenge" + 0.010*"describe" + 0.010*"line" + 0.010*"json" + '
#   '0.010*"extract" + 0.010*"video" + 0.010*"project" + 0.010*"organize" + '
#   '0.010*"large" + 0.010*"request"'),
#  (4,
#   '0.013*"describe" + 0.013*"book" + 0.013*"desperate" + 0.013*"json" + '
#   '0.013*"extract" + 0.013*"video" + 0.013*"project" + 0.013*"organize" + '
#   '0.013*"large" + 0.013*"run"'),
#  (5,
#   '0.013*"describe" + 0.013*"book" + 0.013*"desperate" + 0.013*"json" + '
#   '0.013*"extract" + 0.013*"video" + 0.013*"project" + 0.013*"organize" + '
#   '0.013*"large" + 0.013*"run"'),
#  (6,
#   '0.106*"video" + 0.105*"project" + 0.105*"large" + 0.105*"organize" + '
#   '0.105*"describe" + 0.105*"book" + 0.005*"game" + 0.005*"data" + '
#   '0.005*"scrape" + 0.005*"extract"'),
#  (7,
#   '0.013*"describe" + 0.013*"book" + 0.013*"desperate" + 0.013*"json" + '
#   '0.013*"extract" + 0.013*"video" + 0.013*"project" + 0.013*"organize" + '
#   '0.013*"large" + 0.013*"run"'),
#  (8,
#   '0.176*"extract" + 0.176*"json" + 0.008*"describe" + 0.008*"book" + '
#   '0.008*"desperate" + 0.008*"video" + 0.008*"project" + 0.008*"organize" + '
#   '0.008*"large" + 0.008*"run"'),
#  (9,
#   '0.117*"need" + 0.117*"calculator" + 0.117*"help" + 0.117*"python" + '
#   '0.117*"fail" + 0.006*"organize" + 0.006*"large" + 0.006*"extract" + '
#   '0.006*"video" + 0.006*"json"'),
#  (10,
#   '0.187*"line" + 0.096*"text" + 0.096*"file" + 0.096*"good" + 0.096*"use" + '
#   '0.096*"dash" + 0.005*"describe" + 0.005*"organize" + 0.005*"book" + '
#   '0.005*"work"'),
#  (11,
#   '0.013*"describe" + 0.013*"book" + 0.013*"desperate" + 0.013*"json" + '
#   '0.013*"extract" + 0.013*"video" + 0.013*"project" + 0.013*"organize" + '
#   '0.013*"large" + 0.013*"run"'),
#  (12,
#   '0.075*"specific" + 0.075*"include" + 0.075*"folder" + 0.075*"filename" + '
#   '0.075*"recursively" + 0.075*"edit" + 0.075*"directory" + 0.075*"advice" + '
#   '0.075*"name" + 0.075*"extension"'),
#  (13,
#   '0.081*"work" + 0.081*"make" + 0.081*"random" + 0.081*"walk" + '
#   '0.081*"constrain" + 0.081*"python" + 0.081*"linear" + 0.081*"regression" + '
#   '0.081*"idiot" + 0.004*"dancing"'),
#  (14,
#   '0.013*"describe" + 0.013*"book" + 0.013*"desperate" + 0.013*"json" + '
#   '0.013*"extract" + 0.013*"video" + 0.013*"project" + 0.013*"organize" + '
#   '0.013*"large" + 0.013*"run"'),
#  (15,
#   '0.117*"tip" + 0.117*"job" + 0.117*"developer" + 0.117*"get" + '
#   '0.117*"python" + 0.006*"video" + 0.006*"extract" + 0.006*"json" + '
#   '0.006*"organize" + 0.006*"desperate"'),
#  (16,
#   '0.093*"help" + 0.093*"python" + 0.048*"dataframe" + 0.048*"put" + '
#   '0.048*"run" + 0.048*"request" + 0.048*"open" + 0.048*"datum" + 0.048*"line" '
#   '+ 0.048*"try"'),
#  (17,
#   '0.096*"guess" + 0.096*"number" + 0.096*"read" + 0.096*"error" + '
#   '0.096*"game" + 0.096*"much" + 0.096*"input" + 0.005*"video" + 0.005*"large" '
#   '+ 0.005*"project"'),
#  (18,
#   '0.176*"blockchain" + 0.176*"voting" + 0.008*"json" + 0.008*"extract" + '
#   '0.008*"video" + 0.008*"project" + 0.008*"organize" + 0.008*"large" + '
#   '0.008*"describe" + 0.008*"request"'),
#  (19,
#   '0.013*"describe" + 0.013*"book" + 0.013*"desperate" + 0.013*"json" + '
#   '0.013*"extract" + 0.013*"video" + 0.013*"project" + 0.013*"organize" + '
#   '0.013*"large" + 0.013*"run"')]


# 14. Compute Model Perplexity and Coherence Score
# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
# Perplexity:  -5.775213731318405


# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence= "c_v")
# coherence_lda = coherence_model_lda.get_coherence()
## the above function "get_coherence()" of calculating the Coherence score is not working
# print('\nCoherence Score: ', coherence_lda) 