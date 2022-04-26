# importing the necessary libraries

import streamlit as st

import pickle
import pickleshare
import pygments
import backcall

import pandas as pd
import numpy as np
import six
import re
import tornado
import pywin
import pywin32_bootstrap
from sklearn.feature_extraction.text import TfidfVectorizer

import json

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
stopwords_list ='stopwords-bn.txt'

def cleaned_reviews(input):
    review = review.replace('\n', '') #removing new line 
    input = re.sub('[^\u0980-\u09FF]',' ',str(input)) #removing unnecessary punctuation
    return input

def stopwords_info(filename):

    stp = open(filename,'r',encoding='utf-8').read().split()
    num_of_stopwords = len(stp)
    return stp,num_of_stopwords


def stopword_removal(input,stopwords):
    stp,num_of_stopwords =stopwords_info(stopwords)
    result = input.split()
    reviews = [word.strip() for word in result if word not in stp ]
    reviews =" ".join(reviews)
    return reviews


def process_reviews(input,stopwords,removing_stopwords):
    if removing_stopwords ==False:
        reviews = cleaned_reviews(input)
        
    else:
        reviews = cleaned_reviews(input)
        reviews = stopword_removal(reviews,stopwords)
        
    return reviews    

def calc_trigram_tfidf(reviews):
    """
    This function will return the tf-idf value of the bigram features . 
    
    Args:
        reviews: a list of cleaned reviews   
        
    Returns:
        tfidf: a instance of TfidfVectorizer
        X : Tri-gram Feature Vector (sparse matrix)
    """
    tfidf = TfidfVectorizer(ngram_range=(1,3),use_idf=True,tokenizer=lambda x: x.split()) 
    X = tfidf.fit_transform(reviews)
    
    return tfidf,X

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------

# design elements
st.header('''
NLP Bengali Sentiment Analysis Mini Project
''')

input = st.text_input('Enter your sentence in the Bengali Language')

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# loading the model using pickle
model = open('rr_review_sgd.pkl','rb')
sgd = pickle.load(model)


# predict function

def predict(text):
    processed_review = process_reviews(input,stopwords = stopwords_list,removing_stopwords = True)
    if (len(processed_review))>0:
        # calculate the Unigram Tf-idf feature
        cv,feature_vector = calc_trigram_tfidf(dataset.cleaned) 
        feature = cv.transform([processed_review]).toarray()
        sentiment = sgd.predict(feature)
        score = round(max(sgd.predict_proba(feature).reshape(-1)),2)*100

        if (sentiment ==0):
            st.write('Negative Sentiment and probability is ',score,'%')
        else:
            st.write('Positive Sentiment and probability is ',score,'%')
    else:
        print("This input doesn't contains any bengali Words, thus cannot predict the Sentiment.")
        st.write('Input has no Bangali words. Cannot Predict the Sentiment')

if st.button(label="Submit"):
  try:
    predict(input)
  except:
    pass
else:
  pass
