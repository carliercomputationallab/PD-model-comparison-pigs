# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:45:59 2023
fit LpS for all patients
@author: P70073624
"""

from UFvalues import *
import multiprocessing
import numpy as np
import random
import scipy
import pandas as pd
import time
import sys
import os
import matplotlib.pyplot as plt
from fnmatch import fnmatch

st = time.time()
#%%
def objective(x, predicted_V, Cp, V, cd, Vr, V_fill):
    '''The objective function needed to be minimised'''
        
    t = 240
    predicted_V = rk(t, x, predicted_V, Cp, V, cd, Vr, V_fill)
    
    return predicted_V


#%% OBJECTIVE FUNCTION FOR MINIMISE CANNOT HAVE MORE THAN ONE RETURN OUTPUT, so we 
# repeat the objective function again with just one return
def objective_fn(x, predicted_V, Cp, V, cd, Vr, V_fill):
    '''The objective function needed to be minimised'''
        
    t = 240
    predicted_V = rk(t, x, predicted_V, Cp, V, cd, Vr, V_fill)
    
    return np.sqrt(sum([(a - b)**2 for a, b in zip(V, predicted_V)]))
#%%
#Runge-Kutta
def rk(t, x, predicted_V, Cp, V, cd, Vr, V_fill):

    for timestep in range(0,t): 
                
        v = predicted_V[timestep]
        
        "Apply Runge Kutta Formulas to find next value of y"
        k1 = comdxdt(v, timestep, x,   Cp, cd,  L, V_fill, Vr)
        k2 = comdxdt(v + 0.5  *k1, timestep, x,   Cp, cd,  L,   V_fill, Vr)
        k3 = comdxdt(v + 0.5  *k2, timestep, x,   Cp, cd,  L, V_fill, Vr)
        k4= comdxdt(v + k3, timestep, x,   Cp, cd,  L,   V_fill, Vr)
               
        # Update next value of y
        v = v + ((1 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4))
        predicted_V[timestep+1] = v
        
    return predicted_V


# the differential equations
def comdxdt(v, t, x,  Cp, cd,  L, V_fill, Vr):
 
    LS = x
    
    solutes = ["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"]
        
    af = 16.18 * (1 - np.exp (-0.00077*v))/13.3187
    
    delP = delP0 - ((v - (V_fill+Vr))/490)#consider getting Vfill and Vr from the data file also
    
    #peritoneal concentration gradient
    pr = [phi[i]* RT * (Cp[t][i]-cd[t][i]) for i in range(len(solutes))]
    sigmas_pr = sum(sigma_s*pr)
    sigmal_pr = sum(sigma_l*pr)

    # #volumetric flows across the pores
    J_vC = af*alpha[0]*LS*(delP - sum(pr)) #ml/min
    J_vS = af*alpha[1]*LS*(delP  - sigmas_pr) #ml/min
    J_vL = af*alpha[2]*LS*(delP - sigmal_pr) #ml/min

    dxdt = J_vC + J_vS + J_vL-L
    
    return dxdt


#%%

def multiprocessing_func(pfile):
    Nx = 1
    Cp, V, cd,  L, delP0, V_fill, Vr, predicted_V,predicted_cd, df_cd = input_values(pfile)     
    optimised_values = np.empty(Nx)
    obj_fn = []
    
    for var in range(10):
        
        #Define initial initial_guess
        x0 = random.random()
                
        '''SLSQP optimisation'''
        result = scipy.optimize.minimize(objective_fn, x0, args = (predicted_V, Cp, V, cd, Vr, V_fill),
                method='SLSQP', options = {"maxiter" : 10000, "disp": True})
        
        #gather all optimised values
        optimised_values = np.vstack((optimised_values,result['x'].tolist()))
        predicted_V = objective(result['x'], predicted_V, Cp, V, cd, Vr, V_fill)
        obj_fn.append(result['fun'])
        
    return (optimised_values, obj_fn, predicted_V)
    
#%%

"Get all MTACs for the pig in question"
root = 'patient_files/'
pattern = "*.csv"
patientlist = []
for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            patientlist.append(os.path.join(path, name))

print(patientlist)     


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

L = 0.3 #ml/min 

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

phi = np.array([1, 1, 2*0.96, 1, 1, 1])
#%%

if __name__ == '__main__':
    value = input("Do you want to start the full process? y or n")
    
    if value == 'y':
        pool = multiprocessing.Pool(9)
        result_list = pool.map(multiprocessing_func,patientlist)
        pool.close()
    else:
        sys.exit()
    
    
    et = time.time()
    print('Execution time:', et-st, 'seconds')
    
    
    #%%
    OF = []
    OV = np.empty(len(patientlist),dtype=object)
    UF = np.empty(len(patientlist),dtype=object)
    
    for i in range(len(result_list)):
        OF.append(min(result_list[i][1]))
        OV[i] = result_list[i][0][np.argmin(result_list[i][1])+1]
        UF[i] = result_list[i][2][np.argmin(result_list[i][1])]
        
        
    #%% 
    LPS = pd.DataFrame(columns = ['patient', 'LpS', 'objective fn'], index = range(0, len(patientlist)))
    for i in range(len(patientlist)):
        LPS.loc[i, 'patient'] = patientlist[i]
        LPS.loc[i, 'LpS'] = OV[i]
        LPS.loc[i, 'objective fn'] = OF[i]
    #%%
    # df_OV = df_OV.drop('mean')
    # df_OV.loc['mean'] = df_OV.mean()
    # df_OV.loc['mean'] = df_OV.median()
    # df_OV.loc['mean'] = df_OV[df_OV['objective function']<1e6].mean()
    # df_OV.loc['mean'] = df_OV[df_OV['objective function']==df_OV['objective function'].median()].squeeze()
    # df_OV.loc['mean'] = df_OV[df_OV['objective function']==df_OV['objective function'].min()].squeeze()
       
    #%%
    pfile = patientlist[0]
    Cp, V, cd,  L, delP0, V_fill, Vr, predicted_V,predicted_cd, df_cd = input_values(pfile)
    
    
    predicted_V = objective(LPS.loc[0, 'LpS'], predicted_V, Cp, V, cd, Vr, V_fill)
    fig, ax = plt.subplots(3,2, figsize = (12,18))
    t = 60
    #urea
    df_cd['Urea'].plot( ax = ax[0,0], label = 'data', style = '.')
    ax[0,0].plot(np.arange(t+1),predicted_cd['Urea'], label = 'predicted')
    # ax[0,0].text(0.6, 0.1, f'MTAC = {result["x"][0]:.2f} ml/min', transform=ax[0,0].transAxes)
    ax[0,0].set_title("Urea")
    
    #creatinine
    df_cd['Creatinine'].plot( ax = ax[0,1], style = '.')
    ax[0,1].plot(np.arange(t+1),predicted_cd['Creatinine'])
    # ax[0,1].text(0.6, 0.1, f'MTAC = {result["x"][1]:.2f} ml/min', transform=ax[0,1].transAxes)
    ax[0,1].set_title("Creatinine")
    
    #Sodium
    df_cd['Sodium'].plot( ax = ax[1,0],  style = '.')
    ax[1,0].plot(np.arange(t+1),predicted_cd['Sodium'] )
    # ax[1,0].text(0.6, 0.5, f'MTAC = {result["x"][2]:.2f} ml/min', transform=ax[1,0].transAxes)
    ax[1,0].set_title("Sodium")
    
    #Phosphate
    df_cd['Phosphate'].plot( ax = ax[1,1], style = '.')
    ax[1,1].plot(np.arange(t+1),predicted_cd['Phosphate'] )
    # ax[1,1].text(0.6, 0.1, f'MTAC = {result["x"][3]:.2f} ml/min', transform=ax[1,1].transAxes)
    ax[1,1].set_title("Phosphate")
    
    #Glucose
    df_cd['Glucose'].plot( ax = ax[2,0], style = '.')
    ax[2,0].plot(np.arange(t+1),predicted_cd['Glucose'])
    # ax[2,0].text(0.6, 0.5, f'MTAC = {result["x"][4]:.4f} ml/min', transform=ax[2,0].transAxes)
    ax[2,0].set_title("Glucose")
    
    #Potassium
    df_cd['Potassium'].plot( ax = ax[2,1], style = '.')
    ax[2,1].plot(np.arange(t+1),predicted_cd['Potassium'])
    # ax[2,1].text(0.6, 0.1, f'MTAC = {result["x"][5]:.2f} ml/min', transform=ax[2,1].transAxes)
    ax[2,1].set_title("Potassium")
    
    fig.supxlabel("time, min")
    fig.supylabel("Dialysate concentration, mmol")
    plt.suptitle("Model 1 predictions of dialysate concentration")
    plt.subplots_adjust(top=0.88,
                        bottom=0.11,
                        left=0.09,
                        right=0.9,
                        hspace=0.295,
                        wspace=0.215)
    #%%
    
    # x_avg = np.array(df_OV.loc['mean'])
    # res = [0,0,0]
    # for files in os.listdir('patient_files/pig2/session1/'):
    #     if files not in patientlist:
    #         files = "./patient_files/pig2/session1/"+files
            
    #         predicted_cd, Cp, V, df_cd, Vr, V_fill = input_values(files)
    #         V = V*1000
            
    #         residual, predicted_cd, _ = objective(x_avg, predicted_cd, Cp, V, df_cd, Vr, V_fill)
            
    #         res[0] = res[0] + residual
            
    #         
    
    
            
    # for files in os.listdir('patient_files/pig2/session1/'):
    #     if files not in patientlist:
    #         files = "./patient_files/pig2/session1/"+files
            
    #         predicted_cd, Cp, V, df_cd, Vr, V_fill = input_values(files)
    #         V = V*1000
            
    #         residual, predicted_cd, _ = objective(x_avg, predicted_cd, Cp, V, df_cd, Vr, V_fill)
    #         # print(residual)
    #         res[1] = res[1] + residual
            
    # for files in os.listdir('patient_files/pig2/session1/'):
    #     if files in patientlist:
    #         files = "./patient_files/pig2/session1/"+files
            
    #         predicted_cd, Cp, V, df_cd, Vr, V_fill = input_values(files)
    #         V = V*1000
            
    #         residual, predicted_cd, _ = objective(x_avg, predicted_cd, Cp, V, df_cd, Vr, V_fill)
    #         print(residual)
            
    #         res[2] = res[2] + residual
            
    # print(res)  