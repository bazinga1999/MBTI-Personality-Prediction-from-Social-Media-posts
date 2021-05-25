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
from sklearn.model_selection import GridSearchCV

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

def load_w2v(name):
  with open('W2V_features/'+name+'.npy','rb') as f:
    return np.load(f)


X_train_w2v_IE = load_w2v('X_train_w2v_IE')
X_test_w2v_IE = load_w2v('X_test_w2v_IE')
y_train_w2v_IE = load_w2v('y_train_w2v_IE')
y_test_w2v_IE = load_w2v('y_test_w2v_IE')

mlp_gs = MLPClassifier()
parameter_space = {
    'activation': ['tanh', 'relu', 'logistic'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
clf.fit(X_train_w2v_IE, y_train_w2v_IE)

print("Best Params: ")
print(clf.best_params_)
print()
print(clf.best_estimator_)
print()
grid_predictions = clf.predict(X_test_w2v_IE)
# print classification report
evaluate_D2(y_test_w2v_IE, grid_predictions)



