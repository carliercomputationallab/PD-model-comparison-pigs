

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

#%%
def objective_fn(x, predicted_cd, Cp, V, df_cd, Vr, V_fill):
    '''The objective function needed to be minimised'''
    
    
    t = 240

    predicted_cd, jv = rk(t, x, predicted_cd, Cp, V, df_cd, Vr, V_fill)

    
    return np.sqrt(((df_cd-predicted_cd.loc[df_cd.index])**2).sum(axis = 0))
#%%
#Runge-Kutta
def rk(t, x, predicted_cd, Cp, V, df_cd, Vr, V_fill):

    Jv = []
    
    for timestep in range(0,t): 
        
        cd = predicted_cd.loc[timestep]
        
        "Apply Runge Kutta Formulas to find next value of y"
        k1, v1 = comdxdt(cd, timestep, x,  predicted_cd, Cp, V, df_cd, Vr, V_fill)
        k2, v2 = comdxdt(cd + 0.5  *k1, timestep, x,  predicted_cd, Cp, V, df_cd, Vr, V_fill)
        k3, v3 = comdxdt(cd + 0.5  *k2, timestep, x,  predicted_cd, Cp, V, df_cd, Vr, V_fill)
        k4, v4 = comdxdt(cd + k3, timestep, x,  predicted_cd, Cp, V, df_cd, Vr, V_fill)
        
        # Update next value of y
        cd = cd + (1 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
        Jv.append( (1 / 6.0)*(v1 + 2 * v2 + 2 * v3 + v4))
        #print(UF)
        predicted_cd.loc[timestep+1] = cd
        
        
    return (predicted_cd, Jv)


# the differential equations
def comdxdt(cd, t, x, predicted_cd, Cp, V, df_cd, Vr, V_fill):
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
    PS_s = x[0:6] * 0.998 * abya0_s #fraction small pore surface area - Rippe, A THREE-PORE MODEL OF PERITONEAL TRANSPORT table 1
    # print(PS_s)
    #The current albumin value is from Joost.
    #MTAC for large pores
    PS_l = x[0:6] * 0.002 *abya0_l# Ref: two pore model-oberg,rippe, table 1, A0L/A0 value
    
    af = 16.18 * (1 - np.exp (-0.00077*V[t]))/13.3187
    
    delP = delP0 + ((V[t] - (V_fill+Vr))/490)#consider getting Vfill and Vr from the data file also
    #peritoneal concentration gradient
   
    
    pr = [RT * (Cp[t][i]-cd[i]) for i in range(len(solutes))]
    sigmas_pr = sum([sigma_s[i]*pr[i] for i in range(len(solutes))])
    sigmal_pr = sum([sigma_l[i]*pr[i] for i in range(len(solutes))])
    # print(pr, sigmas_pr)

    # #print("pr", pr, sum(sigma_s*pr.ravel()))
    # #volumetric flows across the pores
    J_vC = af*alpha[0]*LS*(delP - sum(pr)) #ml/min
    J_vS = af*alpha[1]*LS*(delP  - sigmas_pr) #ml/min
    J_vL = af*alpha[2]*LS*(delP - sigmal_pr) #ml/min
    # print(J_vC, J_vS, J_vL)

    # #Peclet numbers
    Pe_s = np.array([J_vS  * (1 - sigma_s[i])/(af*PS_s[i]) for i in range(len(solutes))])
    Pe_l = np.array([J_vL  * (1 - sigma_l[i])/(af*PS_l[i]) for i in range(len(solutes))])
    # print(Pe_s,Pe_l)
    
    # #solute flow rate
    J_sS = (J_vS*(1-sigma_s)*(Cp[t]-cd*np.exp(-Pe_s))/(1-np.exp(-Pe_s))).ravel()
    J_sL = (J_vL*(1-sigma_l)*(Cp[t]-cd*np.exp(-Pe_l))/(1-np.exp(-Pe_l))).ravel()
    # print(J_sS, J_sL)

    # #print("Js", J_sS+J_sL)
    dxdt = ((J_sS + J_sL)/V[t]-np.array(cd)*(J_vC + J_vS + J_vL)/V[t]).ravel()
    # print(dxdt)
    return (dxdt, J_vC+J_vS+J_vL)

LS = 0.074 #ml/min/mmHg

# fractional pore coefficients
alpha = [0.020, 0.900, 0.080]

delP0 = 22 #mmHg

# constant
RT = 19.3 #mmHg per mmol/l
  
#small pore radius
rs = 43 # Angstrom
#large pore radius
rl = 250

solutes = ["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"]
#radius of molecules
r = np.array([ 2.6, 3.0, 2.3, 2.77, 3.7, 2.8]) #the phosphate radius is approximated from its topological surface area
#for radius, new paper by Oberg - https://journals.sagepub.com/doi/suppl/10.1177/08968608211069232
#constants to calculate sigma
gamma_s = r/rs
gamma_l = r/rl


abya0_s = 1+9/8*gamma_s*np.log(gamma_s)-1.56034*gamma_s+0.528155*gamma_s**2+\
    1.91521*gamma_s**3-2.81903*gamma_s**4+0.270788*gamma_s**5+1.10115*gamma_s**6+ 0.435933*gamma_s**7 #eq 21 two pore Ficoll
abya0_l = 1+9/8*gamma_l*np.log(gamma_l)-1.56034*gamma_l+0.528155*gamma_l**2+\
    1.91521*gamma_l**3-2.81903*gamma_l**4+0.270788*gamma_l**5+1.10115*gamma_l**6+ 0.435933*gamma_l**7

# #Osmotic reflection coefficients
sigma_s = np.zeros(len(solutes))
sigma_l = np.zeros(len(solutes))
sigma = np.zeros(len(solutes))

for i in range(len(solutes)):
    sigma_s[i] = 16/3 * (gamma_s[i])**2 - 20/3 * (gamma_s[i])**3 + 7/3 * (gamma_s[i])**4
    sigma_l[i] = 16/3 * (gamma_l[i])**2 - 20/3 * (gamma_l[i])**3 + 7/3 * (gamma_l[i])**4
    sigma[i] = alpha[0] + alpha[1] * sigma_s[i] + alpha[2] * sigma_l[i]
sse_model7 = pd.DataFrame()
#%%
folder = input("Do you want to start for 7_4 or 6_5? 7 or 6")
if folder == '7':
    folder = '7_4ratio/'
elif folder == '6':
    folder = '6_5ratio/'
for file in map(os.path.basename,glob.glob('./spiderdata/'+folder+'m7_iteration*')):
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
        predicted_cd, Cp, V, df_cd, Vr , V_fill = input_values("./patient_files/pig2/session1/"+pfile)
        V = V*1000
        sse_train[pfile] = objective_fn(x_avg, predicted_cd, Cp, V, df_cd, Vr , V_fill)
    
    # sse in the test set in the same session
    testlist = [files for files in os.listdir('patient_files/pig2/session1/') if files not in trainlist]
    sse_test_same = pd.DataFrame(columns = testlist)
    for pfile in testlist:
        predicted_cd, Cp, V, df_cd,  Vr , V_fill = input_values("./patient_files/pig2/session1/"+pfile) 
        V = V*1000
        sse_test_same[pfile] = objective_fn(x_avg, predicted_cd, Cp, V, df_cd, Vr , V_fill)
    
    
    # sse in the test set in the other session
    testlist_other = [files for files in os.listdir('patient_files/pig2/session2/')]
    sse_test_other = pd.DataFrame(columns = testlist_other)
    for pfile in testlist_other:
        predicted_cd, Cp, V, df_cd, Vr , V_fill = input_values("./patient_files/pig2/session2/"+pfile)   
        V = V*1000
        sse_test_other[pfile] = objective_fn(x_avg, predicted_cd, Cp, V, df_cd, Vr , V_fill)
    
    
    sse_model7 = pd.concat([sse_model7,sse_test_same.mean(axis = 1),sse_test_other.mean(axis = 1),sse_train.mean(axis = 1)], axis = 1)

# keys = ['Test set-same session','SD1', 'Test set-other session','SD2', 'Training set', 'SD3']
# result_m7 = pd.concat([sse_model7[0].mean(axis = 1),sse_model7[0].std(axis = 1),
#                     sse_model7[1].mean(axis = 1),sse_model7[1].std(axis = 1),
#                     sse_model7[2].mean(axis = 1),sse_model7[2].std(axis = 1)], axis = 1, 
#                    keys = keys)
# result_m7.to_csv('./spiderdata/'+folder+'model7_persolute_sse.csv')
