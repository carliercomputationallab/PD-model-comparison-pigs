# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 13:06:09 2023

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

st = time.time()

def objective(x, predicted_cd, Cp, V, df_cd, Vr, V_fill, delP0, L):
    '''The objective function needed to be minimised'''
        
    t = 240
    predicted_cd = rk(t, x, predicted_cd, Cp, V, df_cd, Vr, V_fill, delP0, L)
   
    return predicted_cd

#%% OBJECTIVE FUNCTION FOR MINIMISE CANNOT HAVE MORE THAN ONE RETURN OUTPUT, so we 
# repeat the objective function again with just one return
def objective_fn(x, predicted_cd, Cp, V, df_cd, Vr, V_fill, delP0, L):
    '''The objective function needed to be minimised'''
        
    t = 240
    predicted_cd = rk(t, x, predicted_cd, Cp, V, df_cd, Vr, V_fill, delP0, L)
    
    return sum(np.sqrt(((df_cd-predicted_cd.loc[df_cd.index])**2).sum(axis = 0)))
#%%
#Runge-Kutta
def rk(t, x, predicted_cd, Cp, V, df_cd, Vr, V_fill, delP0, L):
   
    df_OV = pd.read_excel('fitted_X.xlsx').set_index('patient').iloc[:,:-1] #from model 7 train parallel for all patients
    MTAC = df_OV.loc[patientlist[0]] #get the MTAc of the patient in question
        
    for timestep in range(0,t): 
        
        cd = predicted_cd.loc[timestep]
        
        "Time dependent MTAC"
        PS_s = MTAC * 0.998 * abya0_s * (1 + x[0]* np.exp(-timestep/x[1])) 
        PS_l = MTAC * 0.002 * abya0_l * (1 + x[0]* np.exp(-timestep/x[1]))
        
        "Apply Runge Kutta Formulas to find next value of y"
        k1 = comdxdt(cd, timestep, x,  predicted_cd, Cp, V, df_cd, Vr, V_fill, delP0, L, PS_s, PS_l)
        k2 = comdxdt(cd + 0.5  *k1, timestep, x,  predicted_cd, Cp, V, df_cd, Vr, V_fill, delP0, L, PS_s, PS_l)
        k3 = comdxdt(cd + 0.5  *k2, timestep, x,  predicted_cd, Cp, V, df_cd, Vr, V_fill, delP0, L, PS_s, PS_l)
        k4 = comdxdt(cd + k3, timestep, x,  predicted_cd, Cp, V, df_cd, Vr, V_fill, delP0, L, PS_s, PS_l)
        
        # Update next value of y
        cd = cd + (1 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
        
        predicted_cd.loc[timestep+1] = cd
                
    return predicted_cd

# the differential equations
def comdxdt(cd, t, x, predicted_cd, Cp, V, df_cd, Vr, V_fill, delP0, L, PS_s, PS_l):

    solutes = ["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"]
        
    af = 16.18 * (1 - np.exp (-0.00077*V[t]))/13.3187
    
    delP = delP0 - ((V[t] - (V_fill+Vr))/490)#consider getting Vfill and Vr from the data file also
    
    #peritoneal concentration gradient
    pr = [phi[i]*RT * (Cp[t][i]-cd[i]) for i in range(len(solutes))]
    sigmas_pr = sum([phi[i] * sigma_s[i]*pr[i] for i in range(len(solutes))])
    sigmal_pr = sum([phi[i] * sigma_l[i]*pr[i] for i in range(len(solutes))])

    # #volumetric flows across the pores
    J_vC = af*alpha[0]*LS*(delP - sum(pr)) #ml/min
    J_vS = af*alpha[1]*LS*(delP  - sigmas_pr) #ml/min
    J_vL = af*alpha[2]*LS*(delP - sigmal_pr) #ml/min

    # #Peclet numbers
    Pe_s = np.array([J_vS  * (1 - sigma_s[i])/(af*PS_s[i]) for i in range(len(solutes))])
    Pe_l = np.array([J_vL  * (1 - sigma_l[i])/(af*PS_l[i]) for i in range(len(solutes))])
    
    # #solute flow rate
    J_sS = (J_vS*(1-sigma_s)*(Cp[t]-cd*np.exp(-Pe_s))/(1-np.exp(-Pe_s))).ravel()
    J_sL = (J_vL*(1-sigma_l)*(Cp[t]-cd*np.exp(-Pe_l))/(1-np.exp(-Pe_l))).ravel()

    dxdt = ((J_sS + J_sL)/V[t]-np.array(cd)*(J_vC + J_vS + J_vL-L)/V[t]).ravel()

    return dxdt

#%%
"the patient in question"
patientlist= ['P10.csv']

"after fitting all LpS using model 7 - V fitting"
LS = 0.045 #ml/min/mmHg

# fractional pore coefficients
alpha = [0.052, 0.900, 0.048]

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

phi = np.array([1, 1, 2*0.96, 1, 1, 1])
#%%

if __name__ == '__main__':
    value = input("Do you want to start the full process? y or n")
    
    if value == 'y':
        
        "the time dependent term has two parameters that need to be fitted- see rk"
        Nx = 2
        pfile = "./patient_files/pig2/session1/"+patientlist[0]
        Cp, V, cd,  L, delP0, V_fill, Vr, predicted_V,predicted_cd, df_cd = input_values(pfile)
        optimised_values = np.empty(Nx)
        obj_fn = []
        
        for var in range(10):
            
            #Define initial initial_guess
            x0 = np.array(random.sample(range(1, 50), Nx))
                        
            '''SLSQP optimisation'''
            result = scipy.optimize.minimize(objective_fn, x0, args = (predicted_cd, Cp, V, df_cd, Vr, V_fill, delP0, L),
                    method='SLSQP', options = {"maxiter" : 1000, "disp": True})
            
            #gather all optimised values
            optimised_values = np.vstack((optimised_values,result['x'].tolist()))
            obj_fn.append(result['fun'])
    
    
    et = time.time()
    print('Execution time:', et-st, 'seconds')
    
    
    #%%
    OF = []
    OV = np.empty(len(patientlist),dtype=object)
    UF = np.empty(len(patientlist),dtype=object)
    
    for i in range(len(patientlist)):
        OF.append(min(obj_fn))
        OV[i] = optimised_values[np.argmin(obj_fn)+1]
        
        
        
    #%% 
    cols = ['x0', 'x1'] 
    df_OV = pd.DataFrame([arr for arr in OV], columns=cols)  
    df_OV['objective function'] = OF
    ''' Note that the optimisation function sometimes predicts that the 
    dialysate concentration is NaN which makes the error zero. Since that
    is indeed the minimum, those x values will be picked up. It is a 
    better idea to look through the optimised values yourself and pick the 
    suitable value'''
       
    #%%
    pfile = "./patient_files/pig2/session1/"+patientlist[0]
    Cp, V, cd,  L, delP0, V_fill, Vr, predicted_V,predicted_cd, df_cd = input_values(pfile)
    predicted_cd = objective(optimised_values[5], predicted_cd, Cp, V, df_cd, Vr, V_fill, delP0, L)
    fig, ax = plt.subplots(3,2, figsize = (12,18))
    t = 240
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