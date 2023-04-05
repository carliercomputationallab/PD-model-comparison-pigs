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

from fnmatch import fnmatch

"Get all MTACs for the pig in question"
root = 'patient_files/pig2/'
pattern = "*.csv"
patientlist = []
for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            patientlist.append(os.path.join(path, name))
            
"Get MTAC"       
solutes = ["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"]
# patientlist= ['P9.csv', 'P15.csv', 'P11.csv', 'P21.csv', 'P10.csv', 'P23.csv', 'P8.csv']
patient_no = []
MTAC_G = pd.Series(dtype=float) 
for pfile in patientlist:  
          
    t = 240 #min
    p = pfile.split("\\")[1] #to get the file name
    print(p)
    patient_no.append(p)
    df = pd.read_csv(pfile,skiprows = range(0,16), delimiter = "," \
                     , encoding= 'unicode_escape')
    print(df.head())
    '''Plasma solute concentration'''
    df_cp = pd.DataFrame(index=[0, 120, 240],columns= solutes, dtype = float)
    df_cp = df[solutes].iloc[1:4].copy()#blood plasma concentration
    for column in df_cp:
        df_cp[column] = df_cp[column].astype('float64')
    index = pd.Series([0,120,240])
    df_cp = df_cp.set_index([index])
    df_cp = df_cp.interpolate(method = 'index', limit_direction = "both")
    df_cp.loc[:,"Potassium"]=4.0
    df_cp.loc[:, "Creatinine"]  *= 0.001
    # uses the interpolation using the values of the indices and interpolates in both directions
    print(df_cp)
    

    '''dialysate solute concentration'''
    df_cd = pd.DataFrame(columns = solutes,dtype = float)
    df_cd = df[solutes].iloc[10:18].copy()  #dialysate concentration
    for column in df_cd:
        df_cd[column] = df_cd[column].astype('float64')  
    df_cd.loc[:, "Creatinine"]  *= 0.001   
    index = pd.Series([0,10,20,30,60,120,180, 240])
    df_cd = df_cd.set_index([index])
    #using .values here to copy only the values, otherwise it tries to match the indices of df and df_cd and it doesnt work
    df_cd = df_cd.interpolate(method = 'index', limit_direction = "both")
    print(df_cd)

    '''dialysate volume'''
    df_V = pd.read_csv(pfile,skiprows = range(0,45), delimiter = ",", \
                       encoding= 'unicode_escape')[["IP volume T=0 (mL)","IP volume T=240 (mL)"]].iloc[0] # IPV measured from haemoglobin
    df_V=df_V.astype(float)
    print(df_V)
    
    MTAC_G = pd.concat([MTAC_G,df_V.mean()/t*np.log(
        (df_V.loc["IP volume T=0 (mL)"]*(df_cp.mean()-df_cd.loc[0]))/
        (df_V.loc["IP volume T=240 (mL)"]*(df_cp.mean()-df_cd.loc[240])))], axis=1)



"Transpose the dataframe and remove the first empty line and index it with patient name"
MTAC_G = MTAC_G.T
MTAC_G = MTAC_G[1:]
MTAC_G.index = patient_no

for _ in range(0,3):
    #pick x nuber of random samples to be the train list
    trainlist=random.sample(os.listdir('patient_files/pig2/session1'),int(folder))
    
    # #use the predicted garred waniewski predicted mtac
    # df = pd.read_csv('garred_waniewski_predictions.csv', header=(1))
    
    #to ignore inf values when calculating mean/median set them to nan
    MTAC_G = MTAC_G.replace([np.inf, -np.inf], np.nan)
    
    MTAC_G.columns = solutes
       
    #take the median of the mtac values, skipna is default to True so all nan values are ignored while calculating median
    x_G = MTAC_G[MTAC_G.index.isin(trainlist)].mean()
    
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