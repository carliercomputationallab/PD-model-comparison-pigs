# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:35:19 2022

@author: P70073624
"""


from values import *
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
#%%
def objective(x, predicted_cd, Cp, V, df_cd, Vr, V_fill):
    '''The objective function needed to be minimised'''
    
    
    t = 240

    predicted_cd, jv = rk(t, x, predicted_cd, Cp, V, df_cd, Vr, V_fill)

    
    return (sum(np.sqrt(((df_cd-predicted_cd.loc[df_cd.index])**2).sum(axis = 0))), predicted_cd, jv)


#%% OBJECTIVE FUNCTION FOR MINIMISE CANNOT HAVE MORE THAN ONE RETURN OUTPUT, so we 
# repeat the objective function again with just one return
def objective_fn(x, predicted_cd, Cp, V, df_cd, Vr, V_fill):
    '''The objective function needed to be minimised'''
     
    t = 240
    
    "call runge Kutta"
    predicted_cd, _ = rk(t, x, predicted_cd, Cp, V, df_cd, Vr, V_fill)

    return sum(np.sqrt(((df_cd-predicted_cd.loc[df_cd.index])**2).sum(axis = 0)))
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
    
    solutes = ["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"]
    
    #MTAC for small pores
    PS_s = x[0:6] * 0.998 * abya0_s #fraction small pore surface area - Rippe, A THREE-PORE MODEL OF PERITONEAL TRANSPORT table 1

    #MTAC for large pores
    PS_l = x[0:6] * 0.002 *abya0_l# Ref: two pore model-oberg,rippe, table 1, A0L/A0 value
    
    #fraction of peritoneal membrane in contact with the dialysis fluid
    af = 16.18 * (1 - np.exp (-0.00077*V[t]))/13.3187
    
    #hydrostatic pressure difference
    delP = delP0 - ((V[t] - (V_fill+Vr))/490)
    
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

    return (dxdt, J_vC+J_vS+J_vL)


#%% 
"minimisation routine"
def multiprocessing_func(pfile):
    
    Nx = 6 #for MTACs
    pfile = "./patient_files/pig2/session1/"+pfile
    print(pfile)
    predicted_cd, Cp, V, df_cd, Vr, V_fill = input_values(pfile)
    V = V*1000
    optimised_values = np.empty(Nx)
    obj_fn = []
    
    for var in range(10):
        
        #Define initial initial_guess
        x0 = np.array(random.sample(range(1, 50), Nx))
        
        '''SLSQP optimisation'''
        result = scipy.optimize.minimize(objective_fn, x0, args = (predicted_cd, Cp, V, df_cd, Vr, V_fill),
                method='SLSQP', bounds = [(0, 200) for _ in x0],
                options = {"maxiter" : 1000, "disp": True})
        
        #gather all optimised values
        optimised_values = np.vstack((optimised_values,result['x'].tolist()))
        
        #gather UF values across the pores
        _,_, jv = objective(result['x'], predicted_cd, Cp, V, df_cd, Vr, V_fill)
        
        #get total error
        obj_fn.append(result['fun'])
        
    return (optimised_values, obj_fn, jv)
    
#%%
# for a 7:4 training to test split
patientlist=random.sample(os.listdir('patient_files/pig2/session1'),7) #randomly generated using random.sample

# ultrafiltration coefficient
LS = 0.074 #ml/min/mmHg

# fractional pore coefficients
alpha = [0.020, 0.900, 0.080]

# Initital hydrostatic pressure difference
delP0 = 8 #mmHg

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

# lymphatic absorption rate
L = 0.3 #ml/min 

abya0_s = 1+9/8*gamma_s*np.log(gamma_s)-1.56034*gamma_s+0.528155*gamma_s**2+\
    1.91521*gamma_s**3-2.81903*gamma_s**4+0.270788*gamma_s**5+1.10115*gamma_s**6+ 0.435933*gamma_s**7 #eq 21 two pore Ficoll
abya0_l = 1+9/8*gamma_l*np.log(gamma_l)-1.56034*gamma_l+0.528155*gamma_l**2+\
    1.91521*gamma_l**3-2.81903*gamma_l**4+0.270788*gamma_l**5+1.10115*gamma_l**6+ 0.435933*gamma_l**7

#Osmotic reflection coefficients
sigma_s = np.zeros(len(solutes))
sigma_l = np.zeros(len(solutes))
sigma = np.zeros(len(solutes))

for i in range(len(solutes)):
    sigma_s[i] = 16/3 * (gamma_s[i])**2 - 20/3 * (gamma_s[i])**3 + 7/3 * (gamma_s[i])**4
    sigma_l[i] = 16/3 * (gamma_l[i])**2 - 20/3 * (gamma_l[i])**3 + 7/3 * (gamma_l[i])**4
    sigma[i] = alpha[0] + alpha[1] * sigma_s[i] + alpha[2] * sigma_l[i]

# dissociation factor
phi = np.array([1, 1, 2*0.96, 1, 1, 1])
#%%

if __name__ == '__main__':
    value = input("Do you want to start the full process? y or n")
    
    "parallel processing"
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
        UF[i] = result_list[i][2][np.argmin(result_list[i][1])] #collect all volume flow through pores
        
        
    #%% 
    cols = ['MTAC_urea', 'MTAC_crea','MTAC_sodium', 'MTAC_phosphate','MTAC_glu', 'MTAC_potassium'] 
    df_OV = pd.DataFrame([arr for arr in OV], columns=cols)  
    df_OV['objective function'] = OF
    df_OV['L'] = UF
    df_OV.loc['mean'] = df_OV[df_OV['objective function']<1e5].mean()
    
    "PS = PS_s + PS_l"
    PS = df_OV.iloc[0:len(patientlist),:6]*0.998*abya0_s+df_OV.iloc[0:len(patientlist),:6]*0.002*abya0_l
    PS['patient'] = patientlist
    PS.set_index('patient', drop = True, inplace = True)

       
    #%%
    
    "Use this to plot"
    
    # pfile = "./patient_files/pig2/session2/"+patientlist[0]
    # predicted_cd, Cp, V, df_cd, Vr, V_fill = input_values(pfile)
    # V = V*1000
    
    # _ , predicted_cd, _ = objective(df_OV.loc['mean'], predicted_cd, Cp, V, df_cd, Vr, V_fill)
    # fig, ax = plt.subplots(3,2, figsize = (12,18))
    # t = 240
    # #urea
    # df_cd['Urea'].plot( ax = ax[0,0], label = 'data', style = '.')
    # ax[0,0].plot(np.arange(t+1),predicted_cd['Urea'], label = 'predicted')
    # # ax[0,0].text(0.6, 0.1, f'MTAC = {result["x"][0]:.2f} ml/min', transform=ax[0,0].transAxes)
    # ax[0,0].set_title("Urea")
    
    # #creatinine
    # df_cd['Creatinine'].plot( ax = ax[0,1], style = '.')
    # ax[0,1].plot(np.arange(t+1),predicted_cd['Creatinine'])
    # # ax[0,1].text(0.6, 0.1, f'MTAC = {result["x"][1]:.2f} ml/min', transform=ax[0,1].transAxes)
    # ax[0,1].set_title("Creatinine")
    
    # #Sodium
    # df_cd['Sodium'].plot( ax = ax[1,0],  style = '.')
    # ax[1,0].plot(np.arange(t+1),predicted_cd['Sodium'] )
    # # ax[1,0].text(0.6, 0.5, f'MTAC = {result["x"][2]:.2f} ml/min', transform=ax[1,0].transAxes)
    # ax[1,0].set_title("Sodium")
    
    # #Phosphate
    # df_cd['Phosphate'].plot( ax = ax[1,1], style = '.')
    # ax[1,1].plot(np.arange(t+1),predicted_cd['Phosphate'] )
    # # ax[1,1].text(0.6, 0.1, f'MTAC = {result["x"][3]:.2f} ml/min', transform=ax[1,1].transAxes)
    # ax[1,1].set_title("Phosphate")
    
    # #Glucose
    # df_cd['Glucose'].plot( ax = ax[2,0], style = '.')
    # ax[2,0].plot(np.arange(t+1),predicted_cd['Glucose'])
    # # ax[2,0].text(0.6, 0.5, f'MTAC = {result["x"][4]:.4f} ml/min', transform=ax[2,0].transAxes)
    # ax[2,0].set_title("Glucose")
    
    # #Potassium
    # df_cd['Potassium'].plot( ax = ax[2,1], style = '.')
    # ax[2,1].plot(np.arange(t+1),predicted_cd['Potassium'])
    # # ax[2,1].text(0.6, 0.1, f'MTAC = {result["x"][5]:.2f} ml/min', transform=ax[2,1].transAxes)
    # ax[2,1].set_title("Potassium")
    
    # fig.supxlabel("time, min")
    # fig.supylabel("Dialysate concentration, mmol")
    # plt.suptitle("Model 1 predictions of dialysate concentration")
    # plt.subplots_adjust(top=0.88,
    #                     bottom=0.11,
    #                     left=0.09,
    #                     right=0.9,
    #                     hspace=0.295,
    #                     wspace=0.215)
    #%%
    "test set first session"
    x_avg = np.array(df_OV.loc['mean'])
    res = [0,0,0]
    for files in os.listdir('patient_files/pig2/session1/'):
        if files not in patientlist:
            files = "./patient_files/pig2/session1/"+files
            predicted_cd, Cp, V, df_cd, Vr, V_fill = input_values(files)
            V = V*1000
            residual, predicted_cd, _ = objective(x_avg, predicted_cd, Cp, V, df_cd, Vr, V_fill)
            res[0] = res[0] + residual
            
    "test set second session"       
    for files in os.listdir('patient_files/pig2/session2/'):
        if files not in patientlist:
            files = "./patient_files/pig2/session2/"+files
            predicted_cd, Cp, V, df_cd, Vr, V_fill = input_values(files)
            V = V*1000
            residual, predicted_cd, _ = objective(x_avg, predicted_cd, Cp, V, df_cd, Vr, V_fill)
            res[1] = res[1] + residual
    
    "training set"
    for files in os.listdir('patient_files/pig2/session1/'):
        if files in patientlist:
            files = "./patient_files/pig2/session1/"+files            
            predicted_cd, Cp, V, df_cd, Vr, V_fill = input_values(files)
            V = V*1000
            residual, predicted_cd, _ = objective(x_avg, predicted_cd, Cp, V, df_cd, Vr, V_fill)
            res[2] = res[2] + residual
            
    print(res)  