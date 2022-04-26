# importing the necessary libraries

import streamlit as st

import pickle
import pickleshare
import pygments
import backcall
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import six
import re
import tornado
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model._stochastic_gradient
from sklearn.linear_model import SGDClassifier
import json
from sklearn.model_selection import train_test_split

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
stopwords_list ='stopwords-bn.txt'

def cleaned_reviews(input):
    input = input.replace('\n', '') #removing new line 
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
    tfidf = TfidfVectorizer(ngram_range=(1,3),use_idf=True,tokenizer=lambda x: x.split()) 
    X = tfidf.fit_transform(reviews)
    
    return tfidf,X

def label_encoding(sentiment,bool):
    le = LabelEncoder()
    le.fit(sentiment)
    encoded_labels = le.transform(sentiment)
    labels = np.array(encoded_labels) # Converting into numpy array
    class_names =le.classes_ ## Define the class names again
    if bool == True:
        print("\n\t\t\t===== Label Encoding =====","\nClass Names:-->",le.classes_)
        for i in sample_data:
            print(sentiment[i],' ', encoded_labels[i],'\n')

    return labels  

data = pd.read_excel('bengali.xlsx')
data['cleaned'] = data['Reviews'].apply(process_reviews,stopwords = stopwords_list,removing_stopwords = True)    
data['length'] = data['cleaned'].apply(lambda x:len(x.split()))
dataset = data.loc[data.length>2]
dataset = dataset.reset_index(drop = True)

lables = label_encoding(dataset.Sentiment,False)
# Split the Feature into train and test set

cv,feature_vector = calc_trigram_tfidf(dataset.cleaned)
feature_space=feature_vector
sentiment=lables
X_train,X_test,y_train,y_test = train_test_split(feature_space,sentiment,train_size = 0.8,
                                                  test_size = 0.2,random_state =0)

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
sgd_model = SGDClassifier(loss ='log',penalty='l2', max_iter=10)
sgd_model.fit(X_train,y_train) 

import pickle
# open a file, where you ant to store the data
file = open('rr_review_sgd.pkl', 'wb')

# dump information to that file
pickle.dump(sgd_model, file)

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
