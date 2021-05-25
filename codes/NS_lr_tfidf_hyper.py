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

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve

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


def evaluate_D2(testlabel, y_pred):
  print(classification_report(testlabel,y_pred))
  print()

  cm1 = confusion_matrix(testlabel,y_pred)
  total1=sum(sum(cm1))
  accuracy1=(cm1[0,0]+cm1[1,1])/total1

  sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
  print('Sensitivity : ', sensitivity1 )
  print()

  specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
  print('Specificity : ', specificity1)
  print()

  
  ckappa = cohen_kappa_score(testlabel, y_pred)
  print("Cohen Kappa Score: ", ckappa)
  
  print()
  
  mcc = matthews_corrcoef(testlabel, y_pred)
  print("MCC Score: ", mcc)
  
  print()
  
  roc_auc = roc_auc_score(testlabel, y_pred)
  print("AUC ROC Score: ",roc_auc)
  print()

  avg_precision = average_precision_score(testlabel, y_pred)

  avg_recall_score = recall_score(testlabel, y_pred)

  
  avg_f1_score = f1_score(testlabel, y_pred)
  # Accuracy, Sensitivity, Specificity, Cohen Kappa, MCC, Precision, Recall, F1, Auc_Roc
  return [accuracy1, sensitivity1, specificity1, ckappa, mcc, avg_precision, avg_recall_score, avg_f1_score, roc_auc]


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
X_train_test_tfidf_IE, X_validation_tfidf_IE, y_train_test_tfidf_IE, y_validation_test_tfidf_IE = train_test_split(data_tfidf_IE, target_mapped_df_IE, test_size=validation_size, stratify=target_mapped_df_IE)
print(X_train_test_tfidf_IE.shape, X_validation_tfidf_IE.shape)

print("NS")
X_train_test_tfidf_NS, X_validation_tfidf_NS, y_train_test_tfidf_NS, y_validation_test_tfidf_NS = train_test_split(data_tfidf_NS, target_mapped_df_NS, test_size=validation_size, stratify=target_mapped_df_NS)
print(X_train_test_tfidf_NS.shape, X_validation_tfidf_NS.shape)

print("TF")
X_train_test_tfidf_TF, X_validation_tfidf_TF, y_train_test_tfidf_TF, y_validation_test_tfidf_TF = train_test_split(data_tfidf_TF, target_mapped_df_TF, test_size=validation_size, stratify=target_mapped_df_TF)
print(X_train_test_tfidf_TF.shape, X_validation_tfidf_TF.shape)

print("JP")
X_train_test_tfidf_JP, X_validation_tfidf_JP, y_train_test_tfidf_JP, y_validation_test_tfidf_JP = train_test_split(data_tfidf_JP, target_mapped_df_JP, test_size=validation_size, stratify=target_mapped_df_JP)
print(X_train_test_tfidf_JP.shape, X_validation_tfidf_JP.shape)

"""**Train -> Train-Test Split (85-15%)**"""

test_size = 0.1
X_train_tfidf_IE, X_test_tfidf_IE, y_train_tfidf_IE, y_test_tfidf_IE = train_test_split(X_train_test_tfidf_IE, y_train_test_tfidf_IE, test_size=test_size, stratify=y_train_test_tfidf_IE)
print(X_train_tfidf_IE.shape, X_test_tfidf_IE.shape)

X_train_tfidf_NS, X_test_tfidf_NS, y_train_tfidf_NS, y_test_tfidf_NS = train_test_split(X_train_test_tfidf_NS, y_train_test_tfidf_NS, test_size=test_size, stratify=y_train_test_tfidf_NS)
print(X_train_tfidf_NS.shape, X_test_tfidf_NS.shape)

X_train_tfidf_TF, X_test_tfidf_TF, y_train_tfidf_TF, y_test_tfidf_TF = train_test_split(X_train_test_tfidf_TF, y_train_test_tfidf_TF, test_size=test_size, stratify=y_train_test_tfidf_TF)
print(X_train_tfidf_TF.shape, X_test_tfidf_TF.shape)

X_train_tfidf_JP, X_test_tfidf_JP, y_train_tfidf_JP, y_test_tfidf_JP = train_test_split(X_train_test_tfidf_JP, y_train_test_tfidf_JP, test_size=test_size, stratify=y_train_test_tfidf_JP)
print(X_train_tfidf_JP.shape, X_test_tfidf_JP.shape)



svc_gs = LogisticRegression()


parameter_space = {"C":np.logspace(-3,3,7), 
                   "penalty":["l1","l2", "elasticnet"],
                   "fit_intercept": [True, False],
                   "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
                   }

# parameter_space = {'C': [0.1], 
#               'gamma': [1],
#               'kernel': ['linear'],
#               } 

from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(svc_gs, parameter_space, n_jobs=-1, cv=5)
clf.fit(X_train_test_tfidf_JP, y_train_test_tfidf_JP['JP'])

print("Best Params: ")
print(clf.best_params_)
print()
print(clf.best_estimator_)
print()
grid_predictions = clf.predict(X_validation_tfidf_JP)
# print classification report
evaluate_D2(y_validation_test_tfidf_JP['JP'], grid_predictions)



# def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
#                         n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
#     """
#     Generate 3 plots: the test and training learning curve, the training
#     samples vs fit times curve, the fit times vs score curve.

#     Parameters
#     ----------
#     estimator : estimator instance
#         An estimator instance implementing `fit` and `predict` methods which
#         will be cloned for each validation.

#     title : str
#         Title for the chart.

#     X : array-like of shape (n_samples, n_features)
#         Training vector, where ``n_samples`` is the number of samples and
#         ``n_features`` is the number of features.

#     y : array-like of shape (n_samples) or (n_samples, n_features)
#         Target relative to ``X`` for classification or regression;
#         None for unsupervised learning.

#     axes : array-like of shape (3,), default=None
#         Axes to use for plotting the curves.

#     ylim : tuple of shape (2,), default=None
#         Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

#     cv : int, cross-validation generator or an iterable, default=None
#         Determines the cross-validation splitting strategy.
#         Possible inputs for cv are:

#           - None, to use the default 5-fold cross-validation,
#           - integer, to specify the number of folds.
#           - :term:`CV splitter`,
#           - An iterable yielding (train, test) splits as arrays of indices.

#         For integer/None inputs, if ``y`` is binary or multiclass,
#         :class:`StratifiedKFold` used. If the estimator is not a classifier
#         or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

#         Refer :ref:`User Guide <cross_validation>` for the various
#         cross-validators that can be used here.

#     n_jobs : int or None, default=None
#         Number of jobs to run in parallel.
#         ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
#         ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
#         for more details.

#     train_sizes : array-like of shape (n_ticks,)
#         Relative or absolute numbers of training examples that will be used to
#         generate the learning curve. If the ``dtype`` is float, it is regarded
#         as a fraction of the maximum size of the training set (that is
#         determined by the selected validation method), i.e. it has to be within
#         (0, 1]. Otherwise it is interpreted as absolute sizes of the training
#         sets. Note that for classification the number of samples usually have
#         to be big enough to contain at least one sample from each class.
#         (default: np.linspace(0.1, 1.0, 5))
#     """
#     if axes is None:
#         _, axes = plt.subplots(1, 3, figsize=(20, 5))

#     axes[0].set_title(title)
#     if ylim is not None:
#         axes[0].set_ylim(*ylim)
#     axes[0].set_xlabel("Training examples")
#     axes[0].set_ylabel("Score")

#     train_sizes, train_scores, test_scores, fit_times, _ = \
#         learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
#                        train_sizes=train_sizes,
#                        return_times=True)
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     fit_times_mean = np.mean(fit_times, axis=1)
#     fit_times_std = np.std(fit_times, axis=1)

#     # Plot learning curve
#     axes[0].grid()
#     axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
#                          train_scores_mean + train_scores_std, alpha=0.1,
#                          color="r")
#     axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
#                          test_scores_mean + test_scores_std, alpha=0.1,
#                          color="g")
#     axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
#                  label="Training score")
#     axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
#                  label="Cross-validation score")
#     axes[0].legend(loc="best")


#     return plt

# title = "JP: Learning Curves | Model: SVM | Features: Tf-Idf "
# # new_clf = SVC(C = clf.best_params_['C'], gamma = clf.best_params_['gamma'], kernel = clf.best_params_['kernel'])
# new_clf = SVC()
# fig, axes = plt.subplots(3, 2, figsize=(10, 15))
# plot_learning_curve(new_clf, title, X_train_test_tfidf_JP, y_train_test_tfidf_JP['JP'], axes=axes[:, 1], ylim=(0.7, 1.01), cv=5, n_jobs=-1)

# plt.savefig('JP_svm_tfidf.png')




