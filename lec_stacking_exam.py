
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, fbeta_score
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold


def SCORES(y_val, pred, proba, str=None, cls_type=None) :
    if cls_type == "m" :
        # print("===========Multi Classifier======")
        acc = accuracy_score(y_val, pred)
        f1 = f1_score(y_val, pred, average='macro')
        auc = roc_auc_score(y_val, proba, average='macro', multi_class='ovo')
        print('{} acc {:.4f}  f1 {:.4f}  auc {:.4f}'.format(str, acc, f1, auc))
    else :
        # print("===========Binary Classifier======")
        acc = accuracy_score(y_val, pred)
        f1 = f1_score(y_val, pred)
        auc = roc_auc_score(y_val, proba[:,1])
        print('acc {:.4f}  f1 {:.4f}  auc {:.4f}  {}'.format(acc, f1, auc, str))


# dataset = load_iris()
# df = pd.DataFrame(data=dataset.data,
#                   #columns=dataset.feature_names
#                   columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
#                   )
# clstype = "m"

dataset = load_breast_cancer()
df = pd.DataFrame(data=dataset.data,
    columns=dataset.feature_names
)
clstype = "s"

df["target"] = dataset.target
X_train = df.iloc[: , :-1]
y_train = df.iloc[: , -1]
X_train7 , X_test3, y_train7, y_test3 = train_test_split(X_train, y_train, test_size=0.3, random_state=121)

model1 = RandomForestClassifier(random_state=11)
model2 = SVC(probability=True)
model3 = DecisionTreeClassifier(random_state=11) #LogisticRegression()

model2.fit(X_train7 , y_train7)
pred = model2.predict(X_test3)
proba = model2.predict_proba(X_test3)
SCORES(y_test3, pred,proba, "[SVC] ", cls_type='s')
# def SCORES(y_val, pred, proba, str=None, cls_type=None) :
# f1 0.9717  f1:0.9412
#=================================================================================

fold_test_tot1 = np.zeros((X_train7.shape[0], 1))
fold_test_tot2 = np.zeros((X_train7.shape[0], 1))
fold_test_tot3 = np.zeros((X_train7.shape[0], 1))

X_test3_tot1 = []
X_test3_tot2 = []
X_test3_tot3 = []

skFold = StratifiedKFold(n_splits=5, random_state=111, shuffle=True)
for loop_cnt, (train_fold_idx, val_fold_idx) in enumerate(skFold.split(X_train7, y_train7)):
    #doto....



#===============================================================================================================
xgb = XGBClassifier()
xgb.fit(new_train_data, y_train7)
xgb_pred = xgb.predict(new_test_mean)
xgb_proba = xgb.predict_proba(new_test_mean)
accuracy_score(y_val3, xgb_pred)
SCORES(y_val3, xgb_pred, xgb_proba, "[XGBClassifier] ", cls_type=clstype)

#===============================================================================================================
lgbm = LGBMClassifier()
lgbm.fit(new_train_data, y_train7)
lgbm_pred = lgbm.predict(new_test_mean)
lgbm_proba = lgbm.predict_proba(new_test_mean)
SCORES(y_val3, lgbm_pred, lgbm_proba, "[LGBMClassifier] ", cls_type=clstype)
