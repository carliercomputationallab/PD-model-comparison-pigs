# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 15:59:57 2022
Waniewski model fitting
@author: P70073624
"""

# import cyipopt 
import numpy as np
import os
import pandas as pd
import random
from values import *
from fnmatch import fnmatch

"Get all MTACs for the pig in question"
root = 'patient_files/pig2/'
pattern = "*.csv"
patientlist = []
for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            patientlist.append(os.path.join(path, name))

print(patientlist)     

"Get MTAC"       
solutes = ["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"]
# patientlist= ['P9.csv', 'P15.csv', 'P11.csv', 'P21.csv', 'P10.csv', 'P23.csv', 'P8.csv']
patient_no = []
MTAC_W = pd.Series(dtype=float) 
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
    
    MTAC_W = pd.concat([MTAC_W,df_V.mean()/t*np.log(
        (pow(df_V.loc["IP volume T=0 (mL)"],(1/2))*(df_cp.mean()-df_cd.loc[0]))/
        (pow(df_V.loc["IP volume T=240 (mL)"],(1/2))*(df_cp.mean()-df_cd.loc[240])))],axis = 1)

"Transpose the dataframe and remove the first empty line and index it with patient name"
MTAC_W = MTAC_W.T
MTAC_W = MTAC_W[1:]
MTAC_W.index = patient_no
print(MTAC_W)

"Collect the total error res and solute specific error sse"
patientlist=random.sample(os.listdir('patient_files/pig2/session1'),6)
sse = [] 
res = []

"Test set first session"
for files in os.listdir('patient_files/pig2/session1/'):
    if files not in patientlist:
        
        pfile = "./patient_files/pig2/session1/"+files
        predicted_cd, Cp, V, df_cd, _, _ = input_values(pfile)
        for t in range(1,241):
            predicted_cd.loc[t] = Cp.mean(axis=0)-(Cp.mean(axis=0)-predicted_cd.loc[0])*((V[0]/V[t])**(1-0.5))*np.exp(-MTAC_W.loc[files]/V.mean()*t/1000)
            
        sse.append(sum(np.sqrt(((df_cd-predicted_cd.loc[df_cd.index])**2).sum(axis = 0))))
        
res.append(sum(sse))

"Test set second session"
sse = []
for files in os.listdir('patient_files/pig2/session2/'):
    if files not in patientlist:
        
        pfile = "./patient_files/pig2/session2/"+files
        predicted_cd, Cp, V, df_cd, _, _ = input_values(pfile)
        for t in range(1,241):
            predicted_cd.loc[t] = Cp.mean(axis=0)-(Cp.mean(axis=0)-predicted_cd.loc[0])*((V[0]/V[t])**(1-0.5))*np.exp(-MTAC_W.loc[files]/V.mean()*t/1000)
            
        sse.append(sum(np.sqrt(((df_cd-predicted_cd.loc[df_cd.index])**2).sum(axis = 0))))
        
res.append(sum(sse))

"Training set"
sse = []
for files in os.listdir('patient_files/pig2/session1/'):
    if files in patientlist:
        
        pfile = "./patient_files/pig2/session1/"+files
        predicted_cd, Cp, V, df_cd, _, _ = input_values(pfile)
        for t in range(1,241):
            predicted_cd.loc[t] = Cp.mean(axis=0)-(Cp.mean(axis=0)-predicted_cd.loc[0])*((V[0]/V[t])**(1-0.5))*np.exp(-MTAC_W.loc[files]/V.mean()*t/1000)
            
        sse.append(sum(np.sqrt(((df_cd-predicted_cd.loc[df_cd.index])**2).sum(axis = 0))))
        
res.append(sum(sse))

print(res)

MTAC_W.to_csv('./spiderdata/6_5ratio/model9_iteration3.csv')

        