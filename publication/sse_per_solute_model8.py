# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 12:16:19 2022
Garred analytical solution
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

sse_model8 = pd.DataFrame()
solutes = ["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"]
folder = input("Do you want to start for 7_4 or 6_5? 7 or 6")


for _ in range(0,3):
    #pick x nuber of random samples to be the train list
    trainlist=random.sample(os.listdir('patient_files/pig2/session1'),int(folder))
    
    #use the predicted garred waniewski predicted mtac
    df = pd.read_csv('garred_waniewski_predictions.csv', header=(1))
    
    #to ignore inf values when calculating mean/median set them to nan
    MTAC_G = df.loc[0:29].replace([np.inf, -np.inf], np.nan).set_index(['0'])
    
    MTAC_G.columns = solutes
    
    p = [files.split('.')[0] for files in trainlist]
    
    #take the median of the mtac values, skipna is default to True so all nan values are ignored while calculating median
    x_G = MTAC_G[MTAC_G.index.isin(p)].mean()
    
#%% TEST SET __ SAME SESSION
    
    #rmse per patient
    sse = [] 
    
    #rmse per model
    res = []
    
    # set the test list
    testlist = [files for files in os.listdir('patient_files/pig2/session1/') if files not in trainlist]
    
    # RMSE per solute for each patient 
    sse_test_same = pd.DataFrame(columns = testlist)
    
    for files in testlist:
  
        predicted_cd, Cp, V, df_cd, _, _ = input_values("./patient_files/pig2/session1/"+files)
        for t in df_cd.index:
            predicted_cd.loc[t] = Cp.mean(axis=0)-(1/V[t])*V[0]*(Cp.mean(axis=0)-predicted_cd.loc[0])*np.exp(-x_G*t/V.mean()/1000)
            
        #store RMSE per solute for each patient separately in this list
        sse_test_same[files] = np.sqrt(((df_cd-predicted_cd)**2).sum(axis = 0))    
        
        # sum RMSE per solute for all patients 
        sse.append(sum(np.sqrt(((df_cd-predicted_cd.loc[df_cd.index])**2).sum(axis = 0))))
            
    # sum all RMSE to obtain the total RMSE for the set
    res.append(sum(sse))

#%% TEST SET __ OTHER SESSION    
    # set rmse per patient to zero
    sse = []
    testlist_other = [files for files in os.listdir('patient_files/pig2/session2/')]
    sse_test_other = pd.DataFrame(columns = testlist_other)
    for files in testlist_other:            
        predicted_cd, Cp, V, df_cd, _, _ = input_values("./patient_files/pig2/session2/"+files)
        for t in df_cd.index:
            predicted_cd.loc[t] = Cp.mean(axis=0)-(1/V[t])*V[0]*(Cp.mean(axis=0)-predicted_cd.loc[0])*np.exp(-x_G*t/V.mean()/1000)
        sse_test_other[files] = np.sqrt(((df_cd-predicted_cd)**2).sum(axis = 0))      
        sse.append(sum(np.sqrt(((df_cd-predicted_cd)**2).sum(axis = 0))))            
    res.append(sum(sse))
    
#%% TRAINING SET
    sse = []    
    sse_train = pd.DataFrame(columns = trainlist)
    for files in trainlist:        
        predicted_cd, Cp, V, df_cd, _, _ = input_values("./patient_files/pig2/session1/"+files)
        for t in df_cd.index:
            predicted_cd.loc[t] = Cp.mean(axis=0)-(1/V[t])*V[0]*(Cp.mean(axis=0)-predicted_cd.loc[0])*np.exp(-x_G*t/V.mean()/1000)
            
            # predicted_cd.loc[t] = Cp.mean(axis=0)-(Cp.mean(axis=0)-predicted_cd.loc[0])*((V[0]/V[t])**(1-0.5))*np.exp(-x_W/V.mean()*t/1000)
        sse_train[files] = np.sqrt(((df_cd-predicted_cd)**2).sum(axis = 0))   
        sse.append(sum(np.sqrt(((df_cd-predicted_cd)**2).sum(axis = 0))))        
    res.append(sum(sse))
    
    # print RMSE per model for each set per iteration
    print(res)
    
    # collect RMSE mean per solute for training and both test sets for 3 iterations in one dataframe which goes to  run_all_sse
    sse_model8 = pd.concat([sse_model8,sse_test_same.mean(axis = 1),sse_test_other.mean(axis = 1),sse_train.mean(axis = 1)], axis = 1)