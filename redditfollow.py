# import praw

# reddit = praw.Reddit(
#     client_id = "8WMKe5l_cakWbw",
#     client_secret = "LaUch7Mm0BsIejJplbbv6Y6ajbzJyw",
#     user_agent = "eddie's user_agent for reddit",
# )


# for submission in reddit.subreddit("learnpython").hot(limit=20):
#     print(submission.title)


# Run in python console
import nltk; 
# nltk.download('stopwords')


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

# Import Dataset
#  This version of the dataset contains about 11k newsgroups posts from 20 different topics. 
df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')


# df = pd.read_json('{\"content\":{\"0\": \"From: sfasf\"},\"target\": 31,\"target_names\": \"rec.autos\"}')
# df = pd.read_json('{\"content\":{\"1\": \"From: fxsadfsad\"}, \"target\": 31, \"target_names\": \"misc.forsale\"}')
# print(df.target_names.unique())
# df.head() 

# print(df)
# print(type(df))
pprint(type(df.content.values)) 



# # Remove Emails
# data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
# # Remove new line characters
# data = [re.sub('\s+', ' ', sent) for sent in data]
# # Remove distracting single quotes
# data = [re.sub("\'", "", sent) for sent in data]
# # pprint(data[:1]) 
# # print(data)

data = ['rec.autos' 'comp.sys.mac.hardware' 'comp.graphics' 'sci.space'
 'talk.politics.guns' 'sci.med' 'comp.sys.ibm.pc.hardware'
 'comp.os.ms-windows.misc' 'rec.motorcycles' 'talk.religion.misc'
 'misc.forsale' 'alt.atheism' 'sci.electronics' 'comp.windows.x'
 'rec.sport.hockey' 'rec.sport.baseball' 'soc.religion.christian'
 'talk.politics.mideast' 'talk.politics.misc' 'sci.crypt']

# it produced the following result:
# ['rec.autos' 'comp.sys.mac.hardware' 'rec.motorcycles' 'misc.forsale'
#  'comp.os.ms-windows.misc' 'alt.atheism' 'comp.graphics'
#  'rec.sport.baseball' 'rec.sport.hockey' 'sci.electronics' 'sci.space'
#  'talk.politics.misc' 'sci.med' 'talk.politics.mideast'
#  'soc.religion.christian' 'comp.windows.x' 'comp.sys.ibm.pc.hardware'
#  'talk.politics.guns' 'talk.religion.misc' 'sci.crypt']

# Convert to list
# data = df.content.values.tolist()
# # Remove Emails
# data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
# # Remove new line characters
# data = [re.sub('\s+', ' ', sent) for sent in data]
# # Remove distracting single quotes
# data = [re.sub("\'", "", sent) for sent in data]

# pprint(data[:1])

# I got the following message when I print the above:
# ['From: (wheres my thing) Subject: WHAT car is this!? Nntp-Posting-Host: '
#  'rac3.wam.umd.edu Organization: University of Maryland, College Park Lines: '
#  '15 I was wondering if anyone out there could enlighten me on this car I saw '
#  'the other day. It was a 2-door sports car, looked to be from the late 60s/ '
#  'early 70s. It was called a Bricklin. The doors were really small. In '
#  'addition, the front bumper was separate from the rest of the body. This is '
#  'all I know. If anyone can tellme a model name, engine specs, years of '
#  'production, where this car is made, history, or whatever info you have on '
#  'this funky looking car, please e-mail. Thanks, - IL ---- brought to you by '
#  'your neighborhood Lerxst ---- ']


# After removing the emails and extra spaces, the text still looks messy. It is not ready for the LDA to consume. 
# You need to break down each sentence into a list of words through tokenization, while clearing up all the messy text in the process.

# Gensim’s simple_preprocess is great for this.

# 8. Tokenize words and Clean-up text
# Let’s tokenize each sentence into a list of words, 
# removing punctuations and unnecessary characters altogether.

# def sent_to_words(sentences):
#     for sentence in sentences:
#         yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

# data_words = list(sent_to_words(data))

# print(data_words[:1]) # comment out this part 

# after printing the above, we get the following data
# [['from', 'wheres', 'my', 'thing', 'subject', 'what', 'car', 'is', 'this', 'nntp', 'posting', 'host', 'rac', 
# 'wam', 'umd', 'edu', 'organization', 'university', 'of', 'maryland', 'college', 'park', 'lines', 'was', 'wondering', 
# 'if', 'anyone', 'out', 'there', 'could', 'enlighten', 'me', 'on', 'this', 'car', 'saw', 'the', 'other', 'day', 'it', 
# 'was', 'door', 'sports', 'car', 'looked', 'to', 'be', 'from', 'the', 'late', 'early', 'it', 'was', 'called', 'bricklin', 
# 'the', 'doors', 'were', 'really', 'small', 'in', 'addition', 'the', 'front', 'bumper', 'was', 'separate', 'from', 
# 'the', 'rest', 'of', 'the', 'body', 'this', 'is', 'all', 'know', 'if', 'anyone', 'can', 'tellme', 'model', 'name', 
# 'engine', 'specs', 'years', 'of', 'production', 'where', 'this', 'car', 'is', 'made', 'history', 'or', 'whatever', 
# 'info', 'you', 'have', 'on', 'this', 'funky', 'looking', 'car', 'please', 'mail', 'thanks', 'il', 'brought', 'to', 
# 'you', 'by', 'your', 'neighborhood', 'lerxst']]


# 9.Creating Bigram and Trigram Models
# Bigrams are two words frequently occurring together in the document. Trigrams are 3 words frequently occurring.

# Some examples in our example are: ‘front_bumper’, ‘oil_leak’, ‘maryland_college_park’ etc.

# Build the bigram and trigram models

# bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
# trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  


# Faster way to get a sentence clubbed as a trigram/bigram
# bigram_mod = gensim.models.phrases.Phraser(bigram)
# trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
# print(trigram_mod[bigram_mod[data_words[0]]])

# below is the output:
# ['from', 'wheres', 'my', 'thing', 'subject', 'what', 'car', 'is', 'this', 'nntp_posting_host',
#  'rac_wam_umd_edu', 'organization', 'university', 'of', 
#  'maryland_college_park', 'lines', 'was', 'wondering', 'if', 
#  'anyone', 'out', 'there', 'could', 'enlighten', 'me', 'on', 'this', 
#  'car', 'saw', 'the', 'other', 'day', 'it', 'was', 'door', 'sports', 
#  'car', 'looked', 'to', 'be', 'from', 'the', 'late', 'early', 'it', 
#  'was', 'called', 'bricklin', 'the', 'doors', 'were', 'really', 
#  'small', 'in', 'addition', 'the', 'front_bumper', 'was', 'separate', 
#  'from', 'the', 'rest', 'of', 'the', 'body', 'this', 'is', 'all', 
#  'know', 'if', 'anyone', 'can', 'tellme', 'model', 'name', 'engine', 
#  'specs', 'years', 'of', 'production', 'where', 'this', 'car', 'is', 
#  'made', 'history', 'or', 'whatever', 'info', 'you', 'have', 'on', 
#  'this', 'funky', 'looking', 'car', 'please', 'mail', 'thanks', 'il', 
#  'brought', 'to', 'you', 'by', 'your', 'neighborhood', 'lerxst']

# the difference between print(data_words[:1]) and 
# print(trigram_mod[bigram_mod[data_words[0]]]) 
# is that the second part creates a list instead of a list of a list

# 10. Remove Stopwords, Make Bigrams and Lemmatize

# Define functions for stopwords, bigrams, trigrams and lemmatization
# def remove_stopwords(texts):
#     return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

# def make_bigrams(texts):
#     return [bigram_mod[doc] for doc in texts]

# def make_trigrams(texts):
#     return [trigram_mod[bigram_mod[doc]] for doc in texts]

# def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
#     """https://spacy.io/api/annotation"""
#     texts_out = []
#     for sent in texts:
#         doc = nlp(" ".join(sent)) 
#         texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
#     return texts_out

# Remove Stop Words

# data_words_nostops = remove_stopwords(data_words)

# Form Bigrams

# data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en

# nlp = spacy.load('en', disable=['parser', 'ner'])
# nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv

# data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

# print(data_lemmatized[:1]) # we can comment out the printing step

# [['where', 's', 'thing', 'car', 'nntp_post', 'host', 'rac_wam', 'umd', 'organization',
#  'university', 'maryland_college', 'park', 'line', 'wonder', 'anyone', 'could', 'enlighten', 
#  'car', 'see', 'day', 'door', 'sport', 'car', 'look', 'late', 'early', 'call', 'bricklin', 
#  'door', 'really', 'small', 'addition', 'front_bumper', 'separate', 'rest', 'body', 
#  'know', 'anyone', 'tellme', 'model', 'name', 'engine', 'spec', 'year', 'production',
#   'car', 'make', 'history', 'whatev', 'info', 'funky', 'look', 'car', 'mail', 'thank', 
#   'bring', 'neighborhood', 'lerxst']]


# Create Dictionary
# id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
# texts = data_lemmatized

# Term Document Frequency
# corpus = [id2word.doc2bow(text) for text in texts]

# View
# print(corpus[:1])

# print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])

# the above will print out the following result, indicating how many times each word appears
# esp: car appears 5 times
# [[('addition', 1), ('body', 1), ('bricklin', 1), ('bring', 1), ('call', 1), ('car', 5), ('day', 1), ('door', 2), 
# ('early', 1), ('engine', 1), ('enlighten', 1), ('funky', 1), ('history', 1), ('host', 1), ('info', 1), ('know', 1), 
# ('late', 1), ('lerxst', 1), ('line', 1), ('look', 2), ('mail', 1), ('make', 1), ('model', 1), ('name', 1), 
# ('neighborhood', 1), ('nntp_poste', 1), ('organization', 1), ('park', 1), ('production', 1), ('rac_wam', 1), 
# ('really', 1), ('rest', 1), ('s', 1), ('saw', 1), ('separate', 1), ('small', 1), ('spec', 1), ('sport', 1), 
# ('tellme', 1), ('thank', 1), ('thing', 1), ('umd', 1), ('university', 1), ('where', 1), ('wonder', 1), ('year', 1)]]


# 12. Building the Topic Model

# Build LDA model
# lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
#                                            id2word=id2word,
#                                            num_topics=20, 
#                                            random_state=100,
#                                            update_every=1,
#                                            chunksize=100,
#                                            passes=10,
#                                            alpha='auto',
#                                            per_word_topics=True)


# 13. View the topics in LDA model
# pprint(lda_model.print_topics())
# doc_lda = lda_model[corpus]

# [(0,
#   '0.538*"ax" + 0.083*"_" + 0.043*"max" + 0.023*"internet" + 0.011*"family" + '
#   '0.007*"iran" + 0.007*"fbi" + 0.006*"ai" + 0.006*"intel" + 0.006*"mr"'),
#  (1,
#   '0.067*"proof" + 0.059*"gateway" + 0.055*"michigan" + 0.015*"cooperation" + '
#   '0.009*"n" + 0.008*"cord" + 0.006*"wipe" + 0.005*"in" + 0.005*"declaration" '
#   '+ 0.005*"bk"'),
#  (2,
#   '0.027*"do" + 0.016*"say" + 0.016*"make" + 0.015*"think" + 0.014*"go" + '
#   '0.014*"know" + 0.013*"time" + 0.012*"well" + 0.012*"see" + 0.012*"people"'),
#  (3,
#   '0.029*"state" + 0.025*"law" + 0.020*"people" + 0.020*"issue" + '
#   '0.019*"government" + 0.018*"right" + 0.016*"public" + 0.015*"child" + '
#   '0.013*"accept" + 0.012*"person"'),
#  (4,
#   '0.139*"physical" + 0.056*"direct" + 0.056*"robert" + 0.032*"rob" + '
#   '0.026*"adam" + 0.025*"daughter" + 0.020*"human_being" + 0.013*"ho" + '
#   '0.010*"cmu" + 0.009*"sister"'),
#  (5,
#   '0.061*"season" + 0.047*"nhl" + 0.047*"goal" + 0.044*"scott" + 0.043*"wing" '
#   '+ 0.034*"pen" + 0.030*"van" + 0.029*"islam" + 0.028*"pittsburgh" + '
#   '0.023*"muslim"'),
#  (6,
#   '0.033*"israel" + 0.020*"american" + 0.018*"israeli" + 0.017*"discussion" + '
#   '0.017*"patient" + 0.017*"report" + 0.016*"pin" + 0.015*"security" + '
#   '0.014*"encryption" + 0.014*"attack"'),
#  (7,
#   '0.047*"window" + 0.044*"file" + 0.038*"program" + 0.028*"com" + '
#   '0.025*"driver" + 0.025*"email" + 0.020*"image" + 0.019*"entry" + '
#   '0.018*"advance" + 0.017*"problem"'),
#  (8,
#   '0.076*"line" + 0.070*"organization" + 0.048*"write" + 0.043*"article" + '
#   '0.029*"nntp_poste" + 0.028*"university" + 0.026*"host" + 0.019*"get" + '
#   '0.017*"reply" + 0.015*"m"'),
#  (9,
#   '0.092*"space" + 0.027*"earth" + 0.024*"launch" + 0.022*"moon" + '
#   '0.021*"nasa" + 0.021*"mission" + 0.020*"orbit" + 0.017*"satellite" + '
#   '0.015*"research" + 0.015*"plane"'),
#  (10,
#   '0.075*"team" + 0.074*"game" + 0.045*"play" + 0.041*"win" + 0.039*"year" + '
#   '0.022*"fan" + 0.019*"trade" + 0.018*"score" + 0.017*"mike" + 0.017*"run"'),
#  (11,
#   '0.090*"gun" + 0.038*"president" + 0.029*"police" + 0.029*"weapon" + '
#   '0.028*"firearm" + 0.024*"rsa" + 0.024*"clipper_chip" + 0.022*"trust" + '
#   '0.020*"arm" + 0.020*"cop"'),
#  (12,
#   '0.038*"armenian" + 0.032*"soldier" + 0.031*"greek" + 0.029*"kill" + '
#   '0.028*"village" + 0.024*"turk" + 0.020*"father" + 0.019*"turkish" + '
#   '0.017*"terrorism" + 0.016*"occupy"'),
#  (13,
#   '0.054*"md" + 0.041*"zone" + 0.035*"russia" + 0.024*"rs" + '
#   '0.020*"correction" + 0.018*"rom" + 0.017*"cd_rom" + 0.011*"television" + '
#   '0.008*"tc" + 0.005*"soviet_union"'),
#  (14,
#   '0.027*"system" + 0.022*"use" + 0.017*"drive" + 0.011*"computer" + '
#   '0.011*"also" + 0.010*"information" + 0.010*"include" + 0.010*"card" + '
#   '0.009*"bit" + 0.009*"work"'),
#  (15,
#   '0.057*"god" + 0.038*"evidence" + 0.031*"christian" + 0.026*"reason" + '
#   '0.024*"believe" + 0.019*"faith" + 0.018*"sense" + 0.017*"exist" + '
#   '0.016*"bible" + 0.016*"claim"'),
#  (16,
#   '0.068*"cool" + 0.035*"cycle" + 0.030*"ticket" + 0.027*"nuclear" + '
#   '0.023*"heat" + 0.022*"water" + 0.017*"custom" + 0.015*"netcom" + '
#   '0.014*"thermal" + 0.014*"hot"'),
#  (17,
#   '0.090*"car" + 0.037*"bike" + 0.024*"engine" + 0.022*"notice" + 0.021*"ride" '
#   '+ 0.018*"sc" + 0.017*"drive" + 0.016*"mile" + 0.015*"insurance" + '
#   '0.014*"road"'),
#  (18,
#   '0.246*"key" + 0.132*"chip" + 0.042*"algorithm" + 0.031*"nsa" + '
#   '0.030*"secure" + 0.022*"bit" + 0.020*"secret" + 0.015*"please_respond" + '
#   '0.015*"agency" + 0.006*"phone"'),
#  (19,
#   '0.042*"solution" + 0.041*"video" + 0.028*"mouse" + 0.023*"period" + '
#   '0.022*"dr" + 0.021*"mhz" + 0.021*"corp" + 0.018*"processor" + '
#   '0.017*"generate" + 0.017*"manager"')]


# How to interpret this?

# Topic 0 is a represented as _0.016“car” + 0.014“power” + 0.010“light” + 0.009“drive” + 
# 0.007“mount” + 0.007“controller” + 0.007“cool” + 0.007“engine” + 0.007“back” + ‘0.006“turn”.

# 14. Compute Model Perplexity and Coherence Score

# Compute Perplexity
# print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
# coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
# coherence_lda = coherence_model_lda.get_coherence() # this part might not be working 
# print('\nCoherence Score: ', coherence_lda)


# Visualize the topics
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
# vis

# mallet_path = '/Users/eddiechen/coding/redditProject/mallet-2.0.8/bin/mallet'

# 16. Building LDA Mallet Model

# def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
#     """
#     Compute c_v coherence for various number of topics
#     Parameters:
#     ----------
#     dictionary : Gensim dictionary
#     corpus : Gensim corpus
#     texts : List of input texts
#     limit : Max num of topics
#     Returns:
#     -------
#     model_list : List of LDA topic models
#     coherence_values : Coherence values corresponding to the LDA model with respective number of topics
#     """
#     coherence_values = []
#     model_list = []
#     for num_topics in range(start, limit, step):
#         model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
#         model_list.append(model)
#         coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
#         coherence_values.append(coherencemodel.get_coherence())

#     return model_list, coherence_values


# print("now i am computing coherence values")
# model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=40, step=6)
# model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=4, step=1)
# when start is at 2, limit is at 2 and step is at 1, it works but nothing prints
# print("now i am plotting but not working yet")

# Show graph
# limit=40; start=2; step=6
# limit=4; start=2; step=1
# x = range(start, limit, step)
# plt.plot(x, coherence_values)
# plt.xlabel("Num Topics")
# plt.ylabel("Coherence score")
# plt.legend(("coherence_values"), loc='best')
# plt.show()

# # Print the coherence scores
# print("printing coherence scores")
# for m, cv in zip(x, coherence_values):
#     print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


# Num Topics = 2   has Coherence Value of 0.4451
# Num Topics = 8   has Coherence Value of 0.5943
# Num Topics = 14  has Coherence Value of 0.6208
# Num Topics = 20  has Coherence Value of 0.6438
# Num Topics = 26  has Coherence Value of 0.643
# Num Topics = 32  has Coherence Value of 0.6478
# Num Topics = 38  has Coherence Value of 0.6525


# Select the model and print the topics
# optimal_model = model_list[3]
# model_topics = optimal_model.show_topics(formatted=False)
# pprint(optimal_model.print_topics(num_words=10))