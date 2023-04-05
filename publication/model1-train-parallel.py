from values import *
import multiprocessing
import numpy as np
import random
import scipy
from scipy import optimize
import pandas as pd
import time
import sys
import os
import matplotlib.pyplot as plt

st = time.time()
#%%
def objective(x, predicted_cd, Cp, V, df_cd):
    '''The objective function needed to be minimised'''

    t = 240

    predicted_cd = rk(t, x, predicted_cd, Cp, V, df_cd)

    
    return (sum(np.sqrt(((df_cd-predicted_cd.loc[df_cd.index])**2).sum(axis = 0))), predicted_cd)


#%%
def objective_fn(x, predicted_cd, Cp, V, df_cd):
    '''The objective function needed to be minimised'''

    t = 240

    predicted_cd = rk(t, x, predicted_cd, Cp, V, df_cd)

    
    return sum(np.sqrt(((df_cd-predicted_cd.loc[df_cd.index])**2).sum(axis = 0)))
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
    L = 0.000125 # lymphatic flow rate
    QU = V[t+1] - V[t] + L #UF rate 
    MC = 0 # since SiCo = 0
    Cl = cp if L < 0 else cd #depending on direction of flow
    dxdt = (MTAC/1000 * (fct * cp - cd) + SiCo * QU * MC - L * Cl)/V[t].ravel()
    return dxdt


def multiprocessing_func(pfile):
    "Minimisation routine"
    Nx = 12 # MTAC and fct to be fitted
    pfile = "./patient_files/pig2/session1/"+pfile
    predicted_cd, Cp, V, df_cd, _ , _ = input_values(pfile)
    optimised_values = np.empty(Nx)
    obj_fn = []
    
    for var in range(10):
        
        #Define initial initial_guess
        x0 = np.array(random.sample(range(1, 50), Nx))

        '''SLSQP optimisation'''
        result = scipy.optimize.minimize(objective_fn, x0, args = (predicted_cd, Cp, V, df_cd),
                method='SLSQP', bounds = [(0, 200) for _ in x0],
                options = {"maxiter" : 1000, "disp": True})
        
        #gather all optimised values
        optimised_values = np.vstack((optimised_values,result['x'].tolist()))
        obj_fn.append(result['fun'])
        
    return (optimised_values, obj_fn)
    
# for a 7:4 training to test split
patientlist= random.sample(os.listdir('patient_files/pig2/session1'),7) #randomly generated using random.sample
patientlist= os.listdir('patient_files/pig2/session1')

if __name__ == '__main__':
    
    value = input("Do you want to start the full process? y or n")
    if value == 'y':
        "parallel processing"
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
    
    for i in range(len(result_list)):
        OF.append(min(result_list[i][1]))
        OV[i] = result_list[i][0][np.argmin(result_list[i][1])+1]
        
        
    #%% 
    cols = ['MTAC_urea', 'MTAC_crea','MTAC_sodium', 'MTAC_phosphate','MTAC_glu', 'MTAC_potassium',
            'fct_urea', 'fct_crea','fct_sodium', 'fct_phosphate','fct_glu', 'fct_potassium'] 
    df_OV = pd.DataFrame([arr for arr in OV], columns=cols)  
    df_OV['objective function'] = OF
    df_OV.loc['mean'] = df_OV[df_OV['objective function']<1e5].mean()
    # df_OV.to_csv('./spiderdata/7_4ratio/model1_iteration3.csv')
       
    #%%
    x_avg = np.array(df_OV.loc['mean'])
    res = [0,0,0]
    
    "test set first session"
    for files in os.listdir('patient_files/pig2/session1/'):
        if files not in patientlist:
            files = "./patient_files/pig2/session1/"+files
            predicted_cd, Cp, V, df_cd, _, _ = input_values(files)
            residual, predicted_cd = objective(x_avg, predicted_cd, Cp, V, df_cd)
            res[0] = res[0] + residual
            
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
    
    
    "test set second session"        
    for files in os.listdir('patient_files/pig2/session2/'):
        if files not in patientlist:
            files = "./patient_files/pig2/session2/"+files
            predicted_cd, Cp, V, df_cd, _, _ = input_values(files)
            residual, predicted_cd = objective(x_avg, predicted_cd, Cp, V, df_cd)
            res[1] = res[1] + residual
    
    "training set"
    for files in os.listdir('patient_files/pig2/session1/'):
        if files in patientlist:
            files = "./patient_files/pig2/session1/"+file
            predicted_cd, Cp, V, df_cd, _, _ = input_values(files)
            residual, predicted_cd = objective(x_avg, predicted_cd, Cp, V, df_cd)
            res[2] = res[2] + residual
            
    print(res)  
    