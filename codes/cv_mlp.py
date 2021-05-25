# -*- coding: utf-8 -*-
"""Prutyay-ML-Project-3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17sW2RGHmCj3XV4RwlAin6D9Hh1E4LlWd
"""

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, hstack
import wordcloud
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neural_network import MLPClassifier
from xgboost import plot_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, balanced_accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")
from sklearn.utils import shuffle
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from nltk.stem import PorterStemmer

data = pd.read_table("mbti_1.csv",delimiter=',',header=None).iloc[1:, :]

data.shape


"""# **Pre-Processing**"""

def preprocess_text(df_old, remove_special=True):
    df = df_old.copy()

    # Remove links 
    df[1] = df[1].apply(lambda x: re.sub(r'https?:\/\/.*?[\s+]', '', x.replace("|"," ") + " "))
    
    #Keep the End Of Sentence characters
    df[1] = df[1].apply(lambda x: re.sub(r'\.', '', x + " "))
    df[1] = df[1].apply(lambda x: re.sub(r'\?', '', x + " "))
    df[1] = df[1].apply(lambda x: re.sub(r'!', '', x + " "))
    
    #Strip Punctation
    df[1] = df[1].apply(lambda x: re.sub(r'[\.+]', ".",x))

    #Remove multiple fullstops
    df[1] = df[1].apply(lambda x: re.sub(r'[^\w\s]','',x))

    #Remove Non-words
    df[1] = df[1].apply(lambda x: re.sub(r'[^a-zA-Z\s]','',x))

    #Convert1to lowercase
    df[1] = df[1].apply(lambda x: x.lower())


    #Remove MBTI codes
    for i in ['isfj','isfp', 'infj', 'infp', 'intj', 'intp', 'estp', 'estj', 'esfp', 'esfj', 'enfp', 'enfj', 'entp', 'entj']:
      df[1] = df[1].apply(lambda x: re.sub(i, '', x))

    #Remove multiple letter repeating words
    df[1] = df[1].apply(lambda x: re.sub(r'([a-z])\1{2,}[\s|\w]*','',x)) 

    #Remove very long words
    df[1] = df[1].apply(lambda x: re.sub(r'(\b\w{30,1000})?\b','',x))
    
    rows = df.shape[0]
    for i in range(rows):
      sentence = df.iloc[i,1]
      # Tokenize: Split the sentence into words
      word_list = nltk.word_tokenize(sentence)
      
      

      # Lemmatize list of words and join
      # ps = PorterStemmer()
      # lemmatized_output = ' '.join([ps.stem(w) for w in word_list])
      
      lemmatizer = WordNetLemmatizer()
      lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])

      df.iloc[i,1] = lemmatized_output

    return df

# Preprocessing of entered Text
new_df = preprocess_text(data)

rows=data.shape[0]
for i in range(rows):
  sent = data.iloc[i,1]
  print(sent)
  break

new_df.head()


"""# **Direction 2**- *Taking 4 binary categories seperately (Making an ensemble of models)*

**Making Targets**
"""

def get_types(row):
    t=row[0]

    I = 0; N = 0
    T = 0; J = 0
    
    if (t[0] == 'I'):
      I = 1
    elif (t[0] == 'E'):
      I = 0
    else:
      print('Wrong Personality')
      return
        
    if (t[1] == 'N'):
      N = 1
    elif (t[1] == 'S'):
      N = 0
    else:
      print('Wrong Personality') 
      return
        
    if (t[2] == 'T'):
      T = 1
    elif (t[2] == 'F'):
      T = 0
    else:
      print('Wrong Personality') 
      return
        
    if (t[3] == 'J'):
      J = 1
    elif (t[3] == 'P'):
      J = 0
    else:
      print('Wrong Personality') 
      return 

    return pd.Series( {'IE':I, 'NS':N , 'TF': T, 'JP': J }) 

mapped_data = new_df.copy()
mapped_data = mapped_data.join(mapped_data.apply (lambda row: get_types (row),axis=1))
mapped_data.head(5)

"""**Removing immbalance from ensemble analysis**"""

def remove_imbalance(data, column):
  values = data[column].value_counts()
  # print(values)
  num_values = min(values[0],values[1])
  # print(num_values)
  data = shuffle(data)
  new_data = shuffle(pd.concat([data.loc[data[column] == 1].iloc[:num_values, :], data.loc[data[column] == 0].iloc[:num_values, :]]))
  return new_data

mapped_data_IE = remove_imbalance(mapped_data, "IE")
mapped_data_NS = remove_imbalance(mapped_data, "NS")
mapped_data_TF = remove_imbalance(mapped_data, "TF")
mapped_data_JP = remove_imbalance(mapped_data, "JP")
print(mapped_data_IE['IE'].value_counts())
print(mapped_data_NS['NS'].value_counts())
print(mapped_data_TF['TF'].value_counts())
print(mapped_data_JP['JP'].value_counts())

"""**Extracting Targets**"""

target_mapped_df_IE = mapped_data_IE[["IE", "NS", "TF", "JP"]]
target_mapped_df_NS = mapped_data_NS[["IE", "NS", "TF", "JP"]]
target_mapped_df_TF = mapped_data_TF[["IE", "NS", "TF", "JP"]]
target_mapped_df_JP = mapped_data_JP[["IE", "NS", "TF", "JP"]]

"""**Feature Extraction - 1 (Count Vectorizer)**"""

vect_count_IE = CountVectorizer(stop_words='english') 
data_count_IE =  vect_count_IE.fit_transform(mapped_data_IE[1])

vect_count_NS = CountVectorizer(stop_words='english') 
data_count_NS =  vect_count_NS.fit_transform(mapped_data_NS[1])

vect_count_TF = CountVectorizer(stop_words='english') 
data_count_TF =  vect_count_TF.fit_transform(mapped_data_TF[1])

vect_count_JP = CountVectorizer(stop_words='english') 
data_count_JP =  vect_count_JP.fit_transform(mapped_data_JP[1])

"""**Feature Extraction - 2 (Tf-idf Vectorizer)**"""

# vect_tfidf = TfidfVectorizer()
# data_tfidf = vect_tfidf.fit_transform(new_df[1])

vect_tfidf_IE = TfidfVectorizer()
data_tfidf_IE =  vect_tfidf_IE.fit_transform(mapped_data_IE[1])

vect_tfidf_NS = TfidfVectorizer()
data_tfidf_NS =  vect_tfidf_NS.fit_transform(mapped_data_NS[1])

vect_tfidf_TF = TfidfVectorizer()
data_tfidf_TF =  vect_tfidf_TF.fit_transform(mapped_data_TF[1])

vect_tfidf_JP = TfidfVectorizer()
data_tfidf_JP =  vect_tfidf_JP.fit_transform(mapped_data_JP[1])

"""**Raw Data -> (Train_Test)/Validation Split (80-20%)** #CountVectotrizer"""

validation_size = 0.2
print("IE")
X_train_test_CoVe_IE, X_validation_CoVe_IE, y_train_test_CoVe_IE, y_validation_test_CoVe_IE = train_test_split(data_count_IE, target_mapped_df_IE, test_size=validation_size, stratify=target_mapped_df_IE)
print(X_train_test_CoVe_IE.shape, X_validation_CoVe_IE.shape)

print("NS")
X_train_test_CoVe_NS, X_validation_CoVe_NS, y_train_test_CoVe_NS, y_validation_test_CoVe_NS = train_test_split(data_count_NS, target_mapped_df_NS, test_size=validation_size, stratify=target_mapped_df_NS)
print(X_train_test_CoVe_NS.shape, X_validation_CoVe_NS.shape)

print("TF")
X_train_test_CoVe_TF, X_validation_CoVe_TF, y_train_test_CoVe_TF, y_validation_test_CoVe_TF = train_test_split(data_count_TF, target_mapped_df_TF, test_size=validation_size, stratify=target_mapped_df_TF)
print(X_train_test_CoVe_TF.shape, X_validation_CoVe_TF.shape)

print("JP")
X_train_test_CoVe_JP, X_validation_CoVe_JP, y_train_test_CoVe_JP, y_validation_test_CoVe_JP = train_test_split(data_count_JP, target_mapped_df_JP, test_size=validation_size, stratify=target_mapped_df_JP)
print(X_train_test_CoVe_JP.shape, X_validation_CoVe_JP.shape)

"""**Train -> Train-Test Split (85-15%)**"""

test_size = 0.15
X_train_CoVe_IE, X_test_CoVe_IE, y_train_CoVe_IE, y_test_CoVe_IE = train_test_split(X_train_test_CoVe_IE, y_train_test_CoVe_IE, test_size=test_size, stratify=y_train_test_CoVe_IE)
print(X_train_CoVe_IE.shape, X_test_CoVe_IE.shape)

X_train_CoVe_NS, X_test_CoVe_NS, y_train_CoVe_NS, y_test_CoVe_NS = train_test_split(X_train_test_CoVe_NS, y_train_test_CoVe_NS, test_size=test_size, stratify=y_train_test_CoVe_NS)
print(X_train_CoVe_NS.shape, X_test_CoVe_NS.shape)

X_train_CoVe_TF, X_test_CoVe_TF, y_train_CoVe_TF, y_test_CoVe_TF = train_test_split(X_train_test_CoVe_TF, y_train_test_CoVe_TF, test_size=test_size, stratify=y_train_test_CoVe_TF)
print(X_train_CoVe_TF.shape, X_test_CoVe_TF.shape)

X_train_CoVe_JP, X_test_CoVe_JP, y_train_CoVe_JP, y_test_CoVe_JP = train_test_split(X_train_test_CoVe_JP, y_train_test_CoVe_JP, test_size=test_size, stratify=y_train_test_CoVe_JP)
print(X_train_CoVe_JP.shape, X_test_CoVe_JP.shape)

y_train_CoVe_IE['IE']

train_test_dataset_mapping = {
    "IE": [X_train_CoVe_IE, X_test_CoVe_IE, y_train_CoVe_IE, y_test_CoVe_IE ],
    "NS": [X_train_CoVe_NS, X_test_CoVe_NS, y_train_CoVe_NS, y_test_CoVe_NS],
    "TF": [X_train_CoVe_TF, X_test_CoVe_TF, y_train_CoVe_TF, y_test_CoVe_TF],
    "JP": [X_train_CoVe_JP, X_test_CoVe_JP, y_train_CoVe_JP, y_test_CoVe_JP ]
}
categories = ["IE", "NS", "TF", "JP"]


print("Classifier: MLP")

from sklearn.model_selection import cross_val_score
clf = MLPClassifier()
scores = cross_val_score(clf, X_train_test_CoVe_IE, y_train_test_CoVe_IE['IE'], cv=5)
print("IE Mean Accuracy: ", max(scores))

clf = MLPClassifier()
scores = cross_val_score(clf, X_train_test_CoVe_NS, y_train_test_CoVe_NS['NS'], cv=5)
print("NS Mean Accuracy: ", max(scores))

clf = MLPClassifier()
scores = cross_val_score(clf, X_train_test_CoVe_TF, y_train_test_CoVe_TF['TF'], cv=5)
print("TF Mean Accuracy: ", max(scores))

clf = MLPClassifier()
scores = cross_val_score(clf, X_train_test_CoVe_JP, y_train_test_CoVe_JP['JP'], cv=5)
print("JP Mean Accuracy: ", max(scores))

