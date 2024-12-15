

import statistics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
f

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
data_yuhou = pd.read_excel(r"xxx.xlsx", sheet_name=0, index_col=0)#Introducing the WM disconnection feature of Whole Brain ROI
X = data_yuhou.iloc[:, :-1]
#X=X[[256,257,259,260,262,263,264,267,268,269,270,285,286,287,
#       288,289,290,291,292,293,38,294,297,42,298,44,299,300,301,
#       48,49,50,302,303,304,305,306,307,66,67,68,335,81,82,83,84,
 #      85,87,88,101,102,103,104,106,110,111,112,126,128,129,131,133,
  #     134,135,136,137,140,141,142,146,148,149,150,189,193,195,199,216,
    #   217,218,232,233,234,235,236,237,238,241,253,254,255]]
#this is the RDN network

scaler=StandardScaler()
X=scaler.fit_transform(X)
X_S=scaler.fit_transform(X)
Y = data_yuhou.iloc[:, -1]

datalabels = np.array(Y)
X = np.array(X)
X_S=np.array(X_S)
Y = np.array(Y)
# Store  weighted F1 scores for each cycle

all_weighted_f1_scores = []

for train_index, test_index in cv.split(X, Y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    
    XGB = XGBClassifier(objective='multi:softprob')
   
    parameters = {'max_depth': [1,2,3], 'learning_rate': np.linspace(0.3, 1.70, 5),
                  'min_child_weight':[7,8,9,10,11,12,13],'n_estimators': [30,40,50,60,77,90,100]#'n_estimators': np.linspace(40, 170, 20,dtype=int)
                 }
    clf = GridSearchCV(XGB, parameters, cv=10, n_jobs=60)
    clf.fit(X_train, y_train)
    
    best_params = clf.best_params_
    best_XGB = XGBClassifier(**best_params)
    
    print("Optimal parameter combination：", clf.best_params_)
    print("Optimal model accuracy：", clf.best_score_)
    
    best_XGB.fit(X_train, y_train)
    
    y_pred = best_XGB.predict(X_test)

    
    f1 = f1_score(y_test, y_pred, average=None)
    
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    all_weighted_f1_scores.append(weighted_f1)
    

all_weighted_f1_scores = []  

for train_index, test_index in cv.split(X, Y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    
    XGB = XGBClassifier(objective='multi:softprob')
   
    parameters = {
        'max_depth': [1, 2, 3], 
        'learning_rate': np.linspace(0.3, 1.70, 5),
        'min_child_weight': [7, 8, 9, 10, 11, 12, 13], 
        'n_estimators': [30, 40, 50, 60, 77, 90, 100]
    }
    clf = GridSearchCV(XGB, parameters, cv=10, n_jobs=60)
    clf.fit(X_train, y_train)
    
    best_params = clf.best_params_
    best_XGB = XGBClassifier(**best_params)
    
    print("Optimal parameter combination:", clf.best_params_)
    print("Optimal model accuracy:", clf.best_score_)
    
    best_XGB.fit(X_train, y_train)
    
    y_pred = best_XGB.predict(X_test)
    
    f1 = f1_score(y_test, y_pred, average=None)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    all_weighted_f1_scores.append(weighted_f1)

confidence_interval = statistics.stdev(all_weighted_f1_scores) * 1.96
mean_performance = statistics.mean(all_weighted_f1_scores)
lower_bound = mean_performance - confidence_interval
upper_bound = mean_performance + confidence_interval

print("Confidence Interval Lower Bound:", lower_bound)
print("Confidence Interval Upper Bound:", upper_bound)
print("Mean Performance:", mean_performance)
print("Confidence Interval:", confidence_interval)
