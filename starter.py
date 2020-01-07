# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 13:34:56 2020

@author: OPO068499
"""

import pandas as pd
import numpy as np

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

extractcl=['avg_1', 'avg_0', 'sd_1', 'sd_0',
       'coefficient_of_variance']

def getdist(row):
    dt=fitness_values[fitness_values["type"]==row["event_name"]]
    ftrain["eavg_1"]=dt["avg_1"]
    ftrain["eavg_0"]=dt["avg_0"]
    ftrain["esd_1"]=dt["sd_1"]
    ftrain["esd_0"]=dt["sd_0"] 
    ftrain["ecoefficient_of_variance"]=dt["coefficient_of_variance"]
    
    return 

fitness_values.columns


ftrain = ftrain.apply(getdist, axis=1)
