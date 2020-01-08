# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 13:34:56 2020

@author: OPO068499
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score

train=pd.read_csv(r"D:\24Projects\Patient Drug_Switch\train_data.csv")
test=pd.read_csv(r"D:\24Projects\Patient Drug_Switch\test_data.csv")
htrain=train.head(10)
fitness_values=pd.read_csv(r"D:\24Projects\Patient Drug_Switch\fitness_values.csv")

train_labels=pd.read_csv(r"D:\24Projects\Patient Drug_Switch\train_labels.csv")
test_labels=pd.read_csv(r"D:\24Projects\Patient Drug_Switch\Sample Submission.csv")


len(train["patient_id"].unique())
len(test["patient_id"].unique())


ftrain=train.groupby("patient_id").last().reset_index()
ftest=test.groupby("patient_id").last().reset_index()


def getdist(txt):
    dst="_".join(txt.split("_")[4:])
    if(dst[0]=="_"):
        dst=dst[1:]
    return dst


fitness_values["type"]=fitness_values["feature_name"].apply(getdist)

extractcl=['avg_1', 'avg_0', 'sd_1', 'sd_0','coefficient_of_variance']

def getdist(row):
    dt=fitness_values[fitness_values["type"]==row["event_name"]]
    return dt[extractcl].iloc[0]

ftrain.columns

ftrain[extractcl] = pd.DataFrame([[0] * len(extractcl)], index=ftrain.index)
ftrain[extractcl] = ftrain.apply(getdist, axis=1)


ftest[extractcl] = pd.DataFrame([[0] * len(extractcl)], index=ftest.index)
ftest[extractcl] = ftrain.apply(getdist, axis=1)



ltrain=pd.merge(ftrain, train_labels, on='patient_id', how='outer')
ltrain.columns


colx=['patient_payment', 'avg_1', 'avg_0', 'sd_1', 'sd_0',
       'coefficient_of_variance','event_time']

X_train, X_test, y_train, y_test = train_test_split(
ltrain[colx],ltrain['outcome_flag'], test_size=0.33, random_state=42)




clf=lgb.LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=-1,
               min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
               n_estimators=1000, n_jobs=-1, num_leaves=31, objective=None,
               random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
               subsample=1.0, subsample_for_bin=200000, subsample_freq=0)



clf.fit(X_train,y_train)
pred=clf.predict(X_test)
accuracy_score(y_test, pred)


clf.fit(ltrain[colx],ltrain['outcome_flag'])
pred=clf.predict(ftest[colx])


ftest["outcome_flag"]=pred

ftest["outcome_flag"].value_counts()
ftest[["patient_id","outcome_flag"]].to_excel(r"D:\24Projects\Patient Drug_Switch\output.xlsx",index=False)


































