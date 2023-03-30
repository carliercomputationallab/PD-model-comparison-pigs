# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 16:33:11 2022
Waniewski analytical solution
@author: P70073624
"""


# import cyipopt 
import numpy as np
import os
import glob
# import numdifftools.nd_statsmodels as nd   
import matplotlib.pyplot as plt 
import random
import pandas as pd

from values import *

# instantiate a dataframe that will keep the mean of all RMSE for training, test-same and test-other per iteration
# In the final two models we are always finding the mean with different sets.
sse_model9 = pd.DataFrame()
solutes = ["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"]
#patientlist= ['P9.csv', 'P15.csv', 'P11.csv', 'P21.csv', 'P10.csv', 'P23.csv', 'P8.csv']
folder = input("Do you want to start for 7_4 or 6_5? 7 or 6")

#we are taking an average of 3 iterations for all models
for _ in range(0,3):
    
    # pick randomly among the 11 dataset the training set. Here I am picking 7. The rest will be test set.
    trainlist=random.sample(os.listdir('patient_files/pig2/session1'),int(folder))
    
    # The csv file here was already prepared to calculate MTAC of all solutes.
    # I will separate the Waniewski predictions below
    df = pd.read_csv('garred_waniewski_predictions.csv', header=(1))
    
    #we want to ignore the values that are infinity or not available while calculating mean
    MTAC_W = df.loc[30:59].replace([np.inf, -np.inf], np.nan).set_index(['0'])
    MTAC_W.columns = solutes
    p = [files.split('.')[0] for files in trainlist]
    x_W = MTAC_W[MTAC_W.index.isin(p)].mean(skipna = True)

    #to keep the RMSE values for each patient    
    sse = [] 
    #to keep the sum of all RMSE for that particular set
    res = []
    
    
#%% TEST SET __ SAME SESSION    
    #same for test set in same session
    testlist = [files for files in os.listdir('patient_files/pig2/session1/') if files not in trainlist]
    sse_test_same = pd.DataFrame(columns = testlist)
    for files in testlist:        
        predicted_cd, Cp, V, df_cd, _, _ = input_values("./patient_files/pig2/session1/"+files)
        for t in df_cd.index:
            predicted_cd.loc[t] = Cp.mean(axis=0)-(Cp.mean(axis=0)-predicted_cd.loc[0])*((V[0]/V[t])**(1-0.5))*np.exp(-x_W/V.mean()*t/1000)
            
            # predicted_cd.loc[t] = Cp.mean(axis=0)-(Cp.mean(axis=0)-predicted_cd.loc[0])*((V[0]/V[t])**(1-0.5))*np.exp(-x_W/V.mean()*t/1000)
        sse_test_same[files] = np.sqrt(((df_cd-predicted_cd)**2).sum(axis = 0))    
        sse.append(sum(np.sqrt(((df_cd-predicted_cd)**2).sum(axis = 0))))           
    res.append(sum(sse))

#%% TEST SET __ OTHER SESSION    
    #same for test set in other session
    sse = []
    testlist_other = [files for files in os.listdir('patient_files/pig2/session2/')]
    sse_test_other = pd.DataFrame(columns = testlist_other)
    for files in testlist_other:
            
            predicted_cd, Cp, V, df_cd, _, _ = input_values("./patient_files/pig2/session2/"+files)
            for t in df_cd.index:
                predicted_cd.loc[t] = Cp.mean(axis=0)-(Cp.mean(axis=0)-predicted_cd.loc[0])*((V[0]/V[t])**(1-0.5))*np.exp(-x_W/V.mean()*t/1000)
            sse_test_other[files] = np.sqrt(((df_cd-predicted_cd)**2).sum(axis = 0))      
            sse.append(sum(np.sqrt(((df_cd-predicted_cd)**2).sum(axis = 0))/df_cd.var()))
            
    res.append(sum(sse))


#%% TRAINING SET     
    #for the training list calculate SSE (sse, defined above) as well as RMSE(sse_train)
    sse = []
    sse_train = pd.DataFrame(columns = trainlist)
    for files in trainlist:
            predicted_cd, Cp, V, df_cd, _, _ = input_values("./patient_files/pig2/session1/"+files)
            for t in df_cd.index:
                predicted_cd.loc[t] = Cp.mean(axis=0)-(Cp.mean(axis=0)-predicted_cd.loc[0])*((V[0]/V[t])**(1-0.5))*np.exp(-x_W/V.mean()*t/1000)
            # for each patient we are saving RMSE per solute
            sse_train[files] = np.sqrt(((df_cd-predicted_cd)**2).sum(axis = 0))   
            sse.append(sum(np.sqrt(((df_cd-predicted_cd)**2).sum(axis = 0))/df_cd.var()))
            
    res.append(sum(sse))
    print(res)
    #calculating first the mean and SD of the test-same,test-other and training dataset and
    #adding it in a dataframe per iteration. This dataframe will contain the mean SSE per solute 
    #in the three types of datasets. Each iteration will have three columns 0,1,2 refereing to 
    #test-same,test-other and training dataset respectively
    sse_model9 = pd.concat([sse_model9,sse_test_same.mean(axis = 1),sse_test_other.mean(axis = 1),sse_train.mean(axis = 1)], axis = 1)