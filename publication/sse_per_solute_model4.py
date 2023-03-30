
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 13:20:58 2022
load spydata and collect RMSE per solute
@author: P70073624
"""

# a naive and incomplete demonstration on how to read a *.spydata file
import pickle
import tarfile
import numpy as np
from values import *
import pandas as pd
import os
import glob
# open a .spydata file

def objective_fn(x, predicted_cd, Cp, V, df_cd):
    '''The objective function needed to be minimised'''
    
    
    t = 240

    predicted_cd = rk(t, x, predicted_cd, Cp, V, df_cd)

    
    return np.sqrt(((df_cd-predicted_cd.loc[df_cd.index])**2).sum(axis = 0))
#%%
#Runge-Kutta
def rk(t, x, predicted_cd, Cp, V, df_cd):
    
   
    
    for timestep in range(0,t): 
        
        cd = predicted_cd.loc[timestep]
        
        "Apply Runge Kutta Formulas to find next value of y"
        k1 = compute(cd, timestep, x,  predicted_cd, Cp, V, df_cd)
        k2 = compute(cd + 0.5  *k1, timestep, x,  predicted_cd, Cp, V, df_cd)
        k3 = compute(cd + 0.5  *k2, timestep, x,  predicted_cd, Cp, V, df_cd)
        k4 = compute(cd + k3, timestep, x,  predicted_cd, Cp, V, df_cd)

        # Update next value of y
        cd = cd + (1 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)

        #print(UF)
        predicted_cd.loc[timestep+1] = cd
        
    return predicted_cd


# the differential equations
def compute(cd, t, x, predicted_cd, Cp, V, df_cd):
    '''
    

    Parameters
    ----------
    cd : predicted dialysate concentration
    t : timepoint
        DESCRIPTION.
    x : intial matrix
        x[0:6] = MTAC
        x[6:12] = fct
        x[12:18] = SiCo
        x[18] = QL
    model : 1-6
        DESCRIPTION.

    Returns
    -------
    dxdt : conc gradient
        derivative

    '''
    
    solutes = ["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"]
    cp = Cp[t] # plasma solute concentration at time t
    #see models of graff for explanation    
    
    MTAC = x[0:6]
    fct = x[6:12]
    SiCo = np.array([0] * len(solutes)) #SiCo
    L = x[12] # lymphatic flow rate
    QU = V[t+1] - V[t] + L #UF rate 
    MC = 0
  
   
    Cl = cp if L < 0 else cd

    dxdt = (MTAC/1000 * (fct * cp - cd) + SiCo * QU * MC - L * Cl)/V[t].ravel()
    return dxdt
folder = input("Do you want to start for 7_4 or 6_5? 7 or 6")
if folder == '7':
    folder = '7_4ratio/'
elif folder == '6':
    folder = '6_5ratio/'
sse_model4 = pd.DataFrame()
#%%
for file in map(os.path.basename,glob.glob('./spiderdata/'+folder+'m4_iteration*')):
    print(file)
    tar = tarfile.open('./spiderdata/'+folder+file, "r")
    # extract all pickled files to the current working directory
    tar.extractall()
    extracted_files = tar.getnames()
    for f in extracted_files:
        if f.endswith('.pickle'):
             with open(f, 'rb') as fdesc:
                 data = pickle.loads(fdesc.read())
                 df_OV = data['df_OV']
                 trainlist = data['patientlist']

    for item in os.listdir():
        if item.endswith('npy') or  item.endswith('pickle'):
            os.remove(item)              
#%%                 
    x_avg = np.array(df_OV.loc['mean'])   
    
    solutes = ["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"]
    
    sse_train = pd.DataFrame(columns = trainlist)
    
    # sse in the training list
    for pfile in trainlist:
        predicted_cd, Cp, V, df_cd, _ , _ = input_values("./patient_files/pig2/session1/"+pfile)
        sse_train[pfile] = objective_fn(x_avg, predicted_cd, Cp, V, df_cd)
    
    # sse in the test set in the same session
    testlist = [files for files in os.listdir('patient_files/pig2/session1/') if files not in trainlist]
    sse_test_same = pd.DataFrame(columns = testlist)
    for pfile in testlist:
        predicted_cd, Cp, V, df_cd, _ , _ = input_values("./patient_files/pig2/session1/"+pfile)       
        sse_test_same[pfile] = objective_fn(x_avg, predicted_cd, Cp, V, df_cd)
    
    
    # sse in the test set in the other session
    testlist_other = [files for files in os.listdir('patient_files/pig2/session2/')]
    sse_test_other = pd.DataFrame(columns = testlist_other)
    for pfile in testlist_other:
        predicted_cd, Cp, V, df_cd, _ , _ = input_values("./patient_files/pig2/session2/"+pfile)       
        sse_test_other[pfile] = objective_fn(x_avg, predicted_cd, Cp, V, df_cd)
    
    
    sse_model4 = pd.concat([sse_model4,sse_test_same.mean(axis = 1),sse_test_other.mean(axis = 1),sse_train.mean(axis = 1)], axis = 1)

keys = ['Test set-same session','SD1', 'Test set-other session','SD2', 'Training set', 'SD3']
result_m4 = pd.concat([sse_model4[0].mean(axis = 1),sse_model4[0].std(axis = 1),
                    sse_model4[1].mean(axis = 1),sse_model4[1].std(axis = 1),
                    sse_model4[2].mean(axis = 1),sse_model4[2].std(axis = 1)], axis = 1, 
                   keys = keys)
result_m4.to_csv('./spiderdata/'+folder+'model4_persolute_sse.csv')