# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 13:33:11 2023
Fitting alpha_c by minimising sodium error
It seems like there are more ultrasmall pores to allow more fluid flow
@author: P70073624
"""


from UFvalues import *
import numpy as np
import random
import scipy
import pandas as pd
import time
import os
import matplotlib.pyplot as plt


st = time.time()
#%%
def objective(x, Cp,Cd,  L,V, delP0, V_fill, Vr, predicted_cd, pfile ):
    '''The objective function needed to be minimised'''
    t = 240
    
    predicted_cd = rk(t, x,  Cp,  L, V,  delP0, V_fill, Vr, predicted_cd, pfile )


    return predicted_cd


#%% OBJECTIVE FUNCTION FOR MINIMISE CANNOT HAVE MORE THAN ONE RETURN OUTPUT, so we 
# repeat the objective function again with just one return
def objective_fn(x,  Cp,  Cd,  L, V,  delP0, V_fill, Vr, predicted_cd, pfile ):
    '''The objective function needed to be minimised'''
        
    t = 240
    predicted_cd = rk(t, x,  Cp, L, V,  delP0, V_fill, Vr, predicted_cd, pfile )
    
    return np.sqrt(sum([(a - b)**2 for a, b in zip(Cd.iloc[:,2], predicted_cd.loc[Cd.index,'Sodium'])]))
#%%
#Runge-Kutta
def rk(t, x,  Cp, L, V, delP0, V_fill, Vr, predicted_cd, pfile):
    
    df_OV = pd.read_excel('fitted_X.xlsx').set_index('patient').iloc[:,:-1] 
    #fitted X comes from fitting all files in model 7 -train-parallel.py df_OV dataframe
    MTAC = df_OV.loc[pfile]
    
    PS_s = MTAC * 0.998 * abya0_s 
    PS_l = MTAC * 0.002 * abya0_l
    
    for timestep in range(0,t): 
        
        cd = predicted_cd.loc[timestep]
                
        "Apply Runge Kutta Formulas to find next value of y"
        k1 = comdxdt(cd, timestep,   Cp, L, V, delP0, V_fill, Vr, PS_s, PS_l, x)
        k2 = comdxdt(cd + 0.5  *k1, timestep,   Cp,  L, V,   delP0, V_fill, Vr, PS_s, PS_l, x)
        k3 = comdxdt(cd + 0.5  *k2, timestep,   Cp,  L, V,   delP0, V_fill, Vr, PS_s, PS_l, x)
        k4 = comdxdt(cd + k3, timestep,    Cp,   L, V,   delP0, V_fill, Vr, PS_s, PS_l, x)
        
        # Update next value of y
        cd = cd + ((1 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4))
        
        predicted_cd.loc[timestep+1] = cd
    # print(predicted_cd)
    return predicted_cd


# the differential equations
def comdxdt(cd, t,  Cp,  L, V,  delP0, V_fill, Vr, PS_s, PS_l, x):

    solutes = ["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"]
    
    for i in range(len(solutes)):
        sigma_s[i] = 16/3 * (gamma_s[i])**2 - 20/3 * (gamma_s[i])**3 + 7/3 * (gamma_s[i])**4
        sigma_l[i] = 16/3 * (gamma_l[i])**2 - 20/3 * (gamma_l[i])**3 + 7/3 * (gamma_l[i])**4
        sigma[i] = x + alpha[1] * sigma_s[i] + (0.1-x) * sigma_l[i]
    
        
    af = 16.18 * (1 - np.exp (-0.00077*V[t]))/13.3187
    
    delP = delP0 - ((V[t] - (V_fill+Vr))/490)#consider getting Vfill and Vr from the data file also
    #peritoneal concentration gradient
    
    pr = [phi[i]* RT * (Cp[t][i]-cd[i]) for i in range(len(solutes))]
    sigmas_pr = sum([sigma_s[i]*pr[i] for i in range(len(solutes))])
    sigmal_pr = sum([sigma_l[i]*pr[i] for i in range(len(solutes))])

    # #volumetric flows across the pores
    J_vC = af*x*LpS*(delP - sum(pr)) #ml/min
    J_vS = af*alpha[1]*LpS*(delP  - sigmas_pr) #ml/min
    J_vL = af*(0.1-x)*LpS*(delP - sigmal_pr) #ml/min
    
    # #Peclet numbers
    Pe_s = np.array((J_vS  * (1 - sigma_s)/(af*PS_s)).ravel()) 
    Pe_s[np.isinf(Pe_s)] = 0
    Pe_l = np.array((J_vL  * (1 - sigma_l[i])/(af*PS_l[i])).ravel())

    # #solute flow rate
    J_sS = (J_vS*(1-sigma_s)*(Cp[t]-cd*np.exp(-Pe_s))/(1-np.exp(-Pe_s))).ravel()
    J_sL = (J_vL*(1-sigma_l)*(Cp[t]-cd*np.exp(-Pe_l))/(1-np.exp(-Pe_l))).ravel()
        
    dxdt = ((J_sS + J_sL)/V[t]-cd*(J_vC + J_vS + J_vL-L)/V[t]).ravel()

    return dxdt



#%%

patientlist=['P10.csv']

# fractional pore coefficients
alpha = [0.020, 0.900, 0.08]

A0dx = 25000

LpS = 0.045 # value from model7 -Vfitting

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

phi = np.array([1, 1, 2*0.96, 1, 1, 1])


if __name__ == '__main__':
    value = input("Do you want to start the full process? y or n")
    
    ALPHA = pd.DataFrame(columns = ['patient', 'ALPHA-ultrasmall', 'objective fn'], index = range(0, len(patientlist)))
    
    if value == 'y':
        for i, pfile in enumerate(patientlist):
            file = "./patient_files/pig2/session1/"+pfile
            Cp, V, _,  L, delP0, V_fill, Vr, predicted_V, predicted_cd, Cd = input_values(file)            
            optimised_values = []
            obj_fn = []
               
            for var in range(10):
                #Define initial initial_guess
                x0 = random.random()/10 #sodium PS
                                
                '''SLSQP optimisation'''
                result = scipy.optimize.minimize(objective_fn, x0, args = (Cp,  Cd,  L, V, delP0, V_fill, Vr, predicted_cd, pfile ),
                        method='SLSQP', bounds = [(0,0.1)], options = {"maxiter" : 1000, "disp": True})
                #gather all optimised values
                optimised_values.append(result['x'])
                obj_fn.append(result['fun'])  
#%%
            ALPHA.loc[i, 'patient'] = pfile
            ALPHA.loc[i, 'ALPHA-ultrasmall'] = optimised_values[obj_fn.index(min(obj_fn))]
            ALPHA.loc[i, 'objective fn'] = min(obj_fn)
            '''Note that the optimisation function sometimes predicts that the 
            dialysate concentration is NaN which makes the error zero. Since that
            is indeed the minimum, those alpha values will be picked up. It is a 
            better idea to look through the optimised values yourself and pick the 
            alpha value '''
            
            #%%
            "use for plot"
            # Cp, V, _,  L, delP0, V_fill, Vr, predicted_V, predicted_cd, Cd = input_values(file)
            # predicted_cd = pd.DataFrame(columns= ["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"])
            # predicted_cd.loc[0] = Cd.loc[0]
            # predicted_cd= objective(0.072, Cp, Cd,  L, V, delP0, V_fill, Vr, predicted_cd )
            # fig, ax = plt.subplots(2,3)
            # Cd['Urea'].plot( ax = ax[0,0], label = 'data', style = '.') 
            # ax[0,0].plot(range(241),predicted_cd['Urea'], label = 'fitted')  
            # ax[0,0].set_title('Urea')
            # Cd['Creatinine'].plot( ax = ax[0,1], label = 'data', style = '.') 
            # ax[0,1].plot(range(241),predicted_cd['Creatinine'], label = 'fitted')  
            # ax[0,1].set_title('Creatinine')
            # Cd['Sodium'].plot( ax = ax[0,2], label = 'data', style = '.') 
            # ax[0,2].plot(range(241),predicted_cd['Sodium'], label = 'fitted')  
            # ax[0,2].set_title('Sodium')
            # Cd['Phosphate'].plot( ax = ax[1,0], label = 'data', style = '.') 
            # ax[1,0].plot(range(241),predicted_cd['Phosphate'], label = 'fitted')  
            # ax[1,0].set_title('Phosphate')
            # Cd['Glucose'].plot( ax = ax[1,1], label = 'data', style = '.') 
            # ax[1,1].plot(range(241),predicted_cd['Glucose'], label = 'fitted')
            # ax[1,1].set_title('Glucose')
            # Cd['Potassium'].plot( ax = ax[1,2], label = 'data', style = '.') 
            # ax[1,2].plot(range(241),predicted_cd['Potassium'], label = 'fitted')
            # ax[1,2].set_title('Potassium')
            # ax[0,0].set_ylabel('cd, mmol/L')
            # ax[1,1].set_xlabel('time, min')
            # plt_text = 'ALPHA=1.9366 ml/min/mmHg'#+str(optimised_values[obj_fn.index(min(obj_fn))][0])+' ml/min/mmHg'
            # ax[0,2].text(100, 130, plt_text)
            # plt.legend()
            # plt.suptitle('Sodium Fitting-alpha_c')
            # plt.show()                                                                 
 