# the following is the link to gensim topic modeling example recommended by Colin
# https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#1introduction
# we need to get a client_id and client_secret as a developer, we can follow the link below for more info
# https://docs.google.com/document/d/1t_HyOwUI69MdRo46BZIzOTsAVpI7CjD100QRqiqkGWk/edit
# The source code from gensim
# https://radimrehurek.com/gensim/models/ldamodel.html#gensim.models.ldamodel.LdaState

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
# the below for loop is for scraping the data from reddit
for submission in reddit.subreddit("Psychedelics").hot(limit=numOfTopic):
    print(submission.title) #uncomment to see all the titles
    titleList.append(submission.title)

# a sample output of 20 titles
# Worth it.
# I used to do this as a kid without even knowing.
# The emesh method is my failsafe administration method for both DMT and 5-MeO
# Psychedelics Could Help Treat the 463 Million People Coping with Diabetes
# Alex Grey Necrophiliac Rumors along with allegations of hostile environment at CoSM
# Evil trips
# Struggling to implement lessons learned from tripping
# To appease the gods👐
# shroom for a new tripper
# Great trip playlist
# Need help: Shrooms and Acid in same weekend, is it recommended?
# Meditating
# triipy ar(en)t
# Are you able to give yourself goosebumps?
# “Word of Mouth” 36x48” oil on canvas by me. Enjoy 🤩 @grave.daisy on insta
# Seizure on mushrooms + lingering effects
# Orchestra on psychedelics.
# PharmaTher Expands Patent Portfolio with Filing of US Patent Application for Ketamine and ...
# Blue lotus
# New Research Finds Promise in Psychiatric Use Of Psilocybin; North American Companies ...


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
pprint(data_words) 
print("**************After cleaning up the text **************")
## the sample output below
# [['worth', 'it'],
#  ['used', 'to', 'do', 'this', 'as', 'kid', 'without', 'even', 'knowing'],
#  ['the',
#   'emesh',
#   'method',
#   'is',
#   'my',
#   'failsafe',
#   'administration',
#   'method',
#   'for',
#   'both',
#   'dmt',
#   'and',
#   'meo'],
#  ['psychedelics',
#   'could',
#   'help',
#   'treat',
#   'the',
#   'million',
#   'people',
#   'coping',
#   'with',
#   'diabetes'],
#  ['alex',
#   'grey',
#   'necrophiliac',
#   'rumors',
#   'along',
#   'with',
#   'allegations',
#   'of',
#   'hostile',
#   'environment',
#   'at',
#   'cosm'],
#  ['evil', 'trips'],
#  ['struggling', 'to', 'implement', 'lessons', 'learned', 'from', 'tripping'],
#  ['to', 'appease', 'the', 'gods'],
#  ['shroom', 'for', 'new', 'tripper'],
#  ['great', 'trip', 'playlist'],
#  ['need',
#   'help',
#   'shrooms',
#   'and',
#   'acid',
#   'in',
#   'same',
#   'weekend',
#   'is',
#   'it',
#   'recommended'],
#  ['meditating'],
#  ['triipy', 'ar', 'en'],
#  ['are', 'you', 'able', 'to', 'give', 'yourself', 'goosebumps'],
#  ['word',
#   'of',
#   'mouth',
#   'oil',
#   'on',
#   'canvas',
#   'by',
#   'me',
#   'enjoy',
#   'grave',
#   'daisy',
#   'on',
#   'insta'],
#  ['seizure', 'on', 'mushrooms', 'lingering', 'effects'],
#  ['orchestra', 'on', 'psychedelics'],
#  ['pharmather',
#   'expands',
#   'patent',
#   'portfolio',
#   'with',
#   'filing',
#   'of',
#   'us',
#   'patent',
#   'application',
#   'for',
#   'ketamine',
#   'and'],
#  ['blue', 'lotus'],
#  ['new',
#   'research',
#   'finds',
#   'promise',
#   'in',
#   'psychiatric',
#   'use',
#   'of',
#   'psilocybin',
#   'north',
#   'american',
#   'companies']]



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

## will get the following sample output:
# ['worth', 'it']
# ['used', 'to', 'do', 'this', 'as', 'kid', 'without', 'even', 'knowing']
# ['the', 'emesh', 'method', 'is', 'my', 'failsafe', 'administration', 'method', 'for', 'both', 'dmt', 'and', 'meo']
# ['psychedelics', 'could', 'help', 'treat', 'the', 'million', 'people', 'coping', 'with', 'diabetes']
# ['alex', 'grey', 'necrophiliac', 'rumors', 'along', 'with', 'allegations', 'of', 'hostile', 'environment', 'at', 'cosm']
# ['evil', 'trips']
# ['struggling', 'to', 'implement', 'lessons', 'learned', 'from', 'tripping']
# ['to', 'appease', 'the', 'gods']
# ['shroom', 'for', 'new', 'tripper']
# ['great', 'trip', 'playlist']
# ['need', 'help', 'shrooms', 'and', 'acid', 'in', 'same', 'weekend', 'is', 'it', 'recommended']
# ['meditating']
# ['triipy', 'ar', 'en']
# ['are', 'you', 'able', 'to', 'give', 'yourself', 'goosebumps']
# ['word', 'of', 'mouth', 'oil', 'on', 'canvas', 'by', 'me', 'enjoy', 'grave', 'daisy', 'on', 'insta']
# ['seizure', 'on', 'mushrooms', 'lingering', 'effects']
# ['orchestra', 'on', 'psychedelics']
# ['pharmather', 'expands', 'patent', 'portfolio', 'with', 'filing', 'of', 'us', 'patent', 'application', 'for', 'ketamine', 'and']
# ['blue', 'lotus']
# ['new', 'research', 'finds', 'promise', 'in', 'psychiatric', 'use', 'of', 'psilocybin', 'north', 'american', 'companies']

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
pprint(data_lemmatized[0:])

# below is a sample output:
# [['worth'],
#  ['use', 'kid', 'even', 'know'],
#  ['emesh', 'method', 'failsafe', 'administration', 'method', 'dmt', 'meo'],
#  ['psychedelic', 'help', 'treat', 'people', 'cope', 'diabete'],
#  ['alex',
#   'grey',
#   'necrophiliac',
#   'rumor',
#   'allegation',
#   'hostile',
#   'environment',
#   'cosm'],
#  ['evil', 'trip'],
#  ['struggle', 'implement', 'lesson', 'learn', 'trip'],
#  ['appease', 'god'],
#  ['shroom', 'new', 'tripper'],
#  ['great', 'trip', 'playlist'],
#  ['need', 'help', 'shroom', 'acid', 'weekend', 'recommend'],
#  ['meditate'],
#  ['triipy', 'en'],
#  ['able', 'give', 'goosebump'],
#  ['word', 'mouth', 'oil', 'canvas', 'enjoy', 'grave', 'daisy', 'insta'],
#  ['seizure', 'mushroom', 'linger', 'effect'],
#  ['orchestra', 'psychedelic'],
#  ['pharmather',
#   'expand',
#   'patent',
#   'portfolio',
#   'file',
#   'patent',
#   'application',
#   'ketamine'],
#  ['blue', 'lotus'],
#  ['new',
#   'research',
#   'find',
#   'promise',
#   'psychiatric',
#   'psilocybin',
#   'north',
#   'american',
#   'company']]

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
pprint(corpus)
print("************** The end of the word frequency **************")
# Generate the following frequency list:


# Gensim creates a unique id for each word in the document. The produced corpus shown above is a mapping of (word_id, word_frequency).
# For example, (0, 1) above implies, word id 0 occurs once in the first document. Likewise, word id 1 occurs twice and so on.
# [[(0, 1)],
#  [(1, 1), (2, 1), (3, 1), (4, 1)],
#  [(5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 2)],
#  [(11, 1), (12, 1), (13, 1), (14, 1), (15, 1), (16, 1)],
#  [(17, 1), (18, 1), (19, 1), (20, 1), (21, 1), (22, 1), (23, 1), (24, 1)],
#  [(25, 1), (26, 1)],
#  [(26, 1), (27, 1), (28, 1), (29, 1), (30, 1)],
#  [(31, 1), (32, 1)],
#  [(33, 1), (34, 1), (35, 1)],
#  [(26, 1), (36, 1), (37, 1)],
#  [(13, 1), (34, 1), (38, 1), (39, 1), (40, 1), (41, 1)],
#  [(42, 1)],
#  [(43, 1), (44, 1)],
#  [(45, 1), (46, 1), (47, 1)],
#  [(48, 1), (49, 1), (50, 1), (51, 1), (52, 1), (53, 1), (54, 1), (55, 1)],
#  [(56, 1), (57, 1), (58, 1), (59, 1)],
#  [(15, 1), (60, 1)],
#  [(61, 1), (62, 1), (63, 1), (64, 1), (65, 2), (66, 1), (67, 1)],
#  [(68, 1), (69, 1)],
#  [(33, 1),
#   (70, 1),
#   (71, 1),
#   (72, 1),
#   (73, 1),
#   (74, 1),
#   (75, 1),
#   (76, 1),
#   (77, 1)]]

# Human readable format of corpus (term-frequency)
termFrequencyList = [[(id2word[id], freq) for id, freq in cp] for cp in corpus[0:]]
print("**************** printing hunman readable format of corpus ****************")
for i in range(len(termFrequencyList)):
    print(termFrequencyList[i])

# sample output:
# [('worth', 1)]
# [('even', 1), ('kid', 1), ('know', 1), ('use', 1)]
# [('administration', 1), ('dmt', 1), ('emesh', 1), ('failsafe', 1), ('meo', 1), ('method', 2)]
# [('cope', 1), ('diabete', 1), ('help', 1), ('people', 1), ('psychedelic', 1), ('treat', 1)]
# [('alex', 1), ('allegation', 1), ('cosm', 1), ('environment', 1), ('grey', 1), ('hostile', 1), ('necrophiliac', 1), ('rumor', 1)]
# [('evil', 1), ('trip', 1)]
# [('trip', 1), ('implement', 1), ('learn', 1), ('lesson', 1), ('struggle', 1)]
# [('appease', 1), ('god', 1)]
# [('new', 1), ('shroom', 1), ('tripper', 1)]
# [('trip', 1), ('great', 1), ('playlist', 1)]
# [('help', 1), ('shroom', 1), ('acid', 1), ('need', 1), ('recommend', 1), ('weekend', 1)]
# [('meditate', 1)]
# [('en', 1), ('triipy', 1)]
# [('able', 1), ('give', 1), ('goosebump', 1)]
# [('canvas', 1), ('daisy', 1), ('enjoy', 1), ('grave', 1), ('insta', 1), ('mouth', 1), ('oil', 1), ('word', 1)]
# [('effect', 1), ('linger', 1), ('mushroom', 1), ('seizure', 1)]
# [('psychedelic', 1), ('orchestra', 1)]
# [('application', 1), ('expand', 1), ('file', 1), ('ketamine', 1), ('patent', 2), ('pharmather', 1), ('portfolio', 1)]
# [('blue', 1), ('lotus', 1)]
# [('new', 1), ('american', 1), ('company', 1), ('find', 1), ('north', 1), ('promise', 1), ('psilocybin', 1), ('psychiatric', 1), ('research', 1)]
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

# Print the Keyword in the 10 topics
print("************** Print the keyword in the 20 topics below **************")
pprint(lda_model.print_topics())

doc_lda = lda_model[corpus]  

print("************** Print doc_lda below **************")

# below is a sample output
# [(0,
#   '0.013*"enjoy" + 0.013*"canvas" + 0.013*"effect" + 0.013*"word" + '
#   '0.013*"oil" + 0.013*"mouth" + 0.013*"insta" + 0.013*"grave" + '
#   '0.013*"mushroom" + 0.013*"daisy"'),
#  (1,
#   '0.178*"trip" + 0.178*"evil" + 0.008*"grave" + 0.008*"enjoy" + '
#   '0.008*"effect" + 0.008*"word" + 0.008*"oil" + 0.008*"mouth" + 0.008*"insta" '
#   '+ 0.008*"mushroom"'),
#  (2,
#   '0.076*"diabete" + 0.076*"psychedelic" + 0.076*"people" + 0.076*"help" + '
#   '0.076*"cope" + 0.076*"treat" + 0.076*"kid" + 0.076*"even" + 0.076*"know" + '
#   '0.076*"use"'),
#  (3,
#   '0.152*"able" + 0.152*"give" + 0.152*"goosebump" + 0.007*"grave" + '
#   '0.007*"canvas" + 0.007*"effect" + 0.007*"word" + 0.007*"oil" + '
#   '0.007*"mouth" + 0.007*"insta"'),
#  (4,
#   '0.106*"help" + 0.106*"recommend" + 0.106*"need" + 0.106*"shroom" + '
#   '0.106*"acid" + 0.106*"weekend" + 0.005*"enjoy" + 0.005*"insta" + '
#   '0.005*"daisy" + 0.005*"goosebump"'),
#  (5,
#   '0.178*"blue" + 0.178*"lotus" + 0.008*"word" + 0.008*"oil" + 0.008*"mouth" + '
#   '0.008*"insta" + 0.008*"grave" + 0.008*"enjoy" + 0.008*"daisy" + '
#   '0.008*"linger"'),
#  (6,
#   '0.013*"enjoy" + 0.013*"canvas" + 0.013*"effect" + 0.013*"word" + '
#   '0.013*"oil" + 0.013*"mouth" + 0.013*"insta" + 0.013*"grave" + '
#   '0.013*"mushroom" + 0.013*"daisy"'),
#  (7,
#   '0.214*"worth" + 0.010*"enjoy" + 0.010*"linger" + 0.010*"effect" + '
#   '0.010*"word" + 0.010*"oil" + 0.010*"mouth" + 0.010*"insta" + 0.010*"grave" '
#   '+ 0.010*"mushroom"'),
#  (8,
#   '0.013*"enjoy" + 0.013*"canvas" + 0.013*"effect" + 0.013*"word" + '
#   '0.013*"oil" + 0.013*"mouth" + 0.013*"insta" + 0.013*"grave" + '
#   '0.013*"mushroom" + 0.013*"daisy"'),
#  (9,
#   '0.118*"trip" + 0.118*"playlist" + 0.118*"great" + 0.118*"en" + '
#   '0.118*"triipy" + 0.006*"mouth" + 0.006*"grave" + 0.006*"oil" + '
#   '0.006*"canvas" + 0.006*"word"'),
#  (10,
#   '0.013*"enjoy" + 0.013*"canvas" + 0.013*"effect" + 0.013*"word" + '
#   '0.013*"oil" + 0.013*"mouth" + 0.013*"insta" + 0.013*"grave" + '
#   '0.013*"mushroom" + 0.013*"daisy"'),
#  (11,
#   '0.098*"method" + 0.098*"patent" + 0.050*"pharmather" + 0.050*"ketamine" + '
#   '0.050*"file" + 0.050*"portfolio" + 0.050*"expand" + 0.050*"application" + '
#   '0.050*"emesh" + 0.050*"dmt"'),
#  (12,
#   '0.118*"struggle" + 0.118*"trip" + 0.118*"implement" + 0.118*"learn" + '
#   '0.118*"lesson" + 0.006*"grave" + 0.006*"word" + 0.006*"oil" + 0.006*"mouth" '
#   '+ 0.006*"linger"'),
#  (13,
#   '0.118*"effect" + 0.118*"mushroom" + 0.118*"linger" + 0.118*"seizure" + '
#   '0.118*"meditate" + 0.006*"oil" + 0.006*"en" + 0.006*"insta" + 0.006*"word" '
#   '+ 0.006*"grave"'),
#  (14,
#   '0.178*"appease" + 0.178*"god" + 0.008*"enjoy" + 0.008*"canvas" + '
#   '0.008*"effect" + 0.008*"word" + 0.008*"oil" + 0.008*"mouth" + 0.008*"insta" '
#   '+ 0.008*"grave"'),
#  (15,
#   '0.053*"oil" + 0.053*"word" + 0.053*"hostile" + 0.053*"grey" + 0.053*"cosm" '
#   '+ 0.053*"allegation" + 0.053*"alex" + 0.053*"environment" + 0.053*"rumor" + '
#   '0.053*"mouth"'),
#  (16,
#   '0.152*"tripper" + 0.152*"shroom" + 0.152*"new" + 0.007*"enjoy" + '
#   '0.007*"goosebump" + 0.007*"word" + 0.007*"oil" + 0.007*"mouth" + '
#   '0.007*"insta" + 0.007*"grave"'),
#  (17,
#   '0.013*"enjoy" + 0.013*"canvas" + 0.013*"effect" + 0.013*"word" + '
#   '0.013*"oil" + 0.013*"mouth" + 0.013*"insta" + 0.013*"grave" + '
#   '0.013*"mushroom" + 0.013*"daisy"'),
#  (18,
#   '0.081*"research" + 0.081*"company" + 0.081*"psilocybin" + 0.081*"promise" + '
#   '0.081*"north" + 0.081*"find" + 0.081*"new" + 0.081*"american" + '
#   '0.081*"psychiatric" + 0.004*"enjoy"'),
#  (19,
#   '0.013*"enjoy" + 0.013*"canvas" + 0.013*"effect" + 0.013*"word" + '
#   '0.013*"oil" + 0.013*"mouth" + 0.013*"insta" + 0.013*"grave" + '
#   '0.013*"mushroom" + 0.013*"daisy"')]



# 14. Compute Model Perplexity and Coherence Score
# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
# Perplexity:  -5.834839066793752


# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence= "c_v")
# coherence_lda = coherence_model_lda.get_coherence()
## the above function "get_coherence()" of calculating the Coherence score is not working
# print('\nCoherence Score: ', coherence_lda) 