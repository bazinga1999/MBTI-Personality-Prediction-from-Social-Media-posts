
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

from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from nltk.stem import PorterStemmer


def load_w2v(name):
  with open('W2V_features/'+name+'.npy','rb') as f:
    return np.load(f)


X_train_w2v_IE = load_w2v('X_train_w2v_IE')
X_train_w2v_NS = load_w2v('X_train_w2v_NS')
X_train_w2v_TF = load_w2v('X_train_w2v_TF')
X_train_w2v_JP = load_w2v('X_train_w2v_JP')

X_test_w2v_IE = load_w2v('X_test_w2v_IE')
X_test_w2v_NS = load_w2v('X_test_w2v_NS')
X_test_w2v_TF = load_w2v('X_test_w2v_TF')
X_test_w2v_JP = load_w2v('X_test_w2v_JP')

y_train_w2v_IE = load_w2v('y_train_w2v_IE')
y_train_w2v_NS = load_w2v('y_train_w2v_NS')
y_train_w2v_TF = load_w2v('y_train_w2v_TF')
y_train_w2v_JP = load_w2v('y_train_w2v_JP')

y_test_w2v_IE = load_w2v('y_test_w2v_IE')
y_test_w2v_NS = load_w2v('y_test_w2v_NS')
y_test_w2v_TF = load_w2v('y_test_w2v_TF')
y_test_w2v_JP = load_w2v('y_test_w2v_JP')

print("Classifier: SVC")

from sklearn.model_selection import cross_val_score
clf = MLPClassifier()
scores = cross_val_score(clf, X_train_w2v_IE, y_train_w2v_IE, cv=5, n_jobs=5)
print("IE Mean Accuracy: ", max(scores))

clf = MLPClassifier()
scores = cross_val_score(clf, X_train_w2v_NS, y_train_w2v_NS, cv=5, n_jobs=5)
print("NS Mean Accuracy: ", max(scores))

clf = MLPClassifier()
scores = cross_val_score(clf, X_train_w2v_TF, y_train_w2v_TF, cv=5, n_jobs=5)
print("TF Mean Accuracy: ", max(scores))

clf = MLPClassifier()
scores = cross_val_score(clf, X_train_w2v_JP, y_train_w2v_JP, cv=5, n_jobs=5)
print("JP Mean Accuracy: ", max(scores))