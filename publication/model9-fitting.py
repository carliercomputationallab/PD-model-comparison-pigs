# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 13:52:30 2023
Does fitting f other than 0.5 help???
@author: P70073624
"""

 

# import cyipopt 
import numpy as np
import os
import glob 
import matplotlib.pyplot as plt 
import scipy
from scipy.optimize import minimize
import pandas as pd
from scipy.interpolate import interp1d


from sklearn.linear_model import LinearRegression
#%%
def objective(fsel, i, cp, V, df_cd, cd, predicted_cd):
    "involving all data points to calculate MTAC"
    
    x_W = []
    
    for j in range(0, len(fsel)):
        Y = np.log(pow(V,1-fsel[j])*np.abs(cp[:,j]-cd[:,j])).reshape(-1,1)
        lr = LinearRegression()  # create object for the class
        lr.fit(X, Y)  # perform linear 
        x_W.append(-lr.coef_[0,0]*V.mean())
            
        for t in range(1,241):
            predicted_cd.loc[t,solutes[j]] = cp[:,j].mean()-(cp[:,j].mean()-predicted_cd.loc[0, solutes[j]])*((V[0]/V[t])**(1-fsel[j]))*np.exp(-x_W[j]/V.mean()*t)
    
    x_W = np.array(x_W)
    
    return predicted_cd

def objective3(cp, V, df_cd, cd, predicted_cd):
    "taking only the first and last data points to calculate MTAC keeping f fixed at 0.5"
    
    x_W= V.mean()/240*np.log((pow(V[0],1/2)*(cp.mean(axis = 0)-df_cd.loc[0]))/
                                     (pow(V[240],1/2)*(cp.mean(axis= 0)-df_cd.loc[240])))
       
    for t in range(1,241):
        
        predicted_cd.loc[t] = cp.mean(axis=0)-(cp.mean(axis=0)-predicted_cd.loc[0])*((V[0]/V[t])**(1-0.5))*np.exp(-x_W/V.mean()*t)
    
    return predicted_cd
#%%
def objective_fn(f, i, cp, V, df_cd, cd):
    "fitting f"
    print(pow(V,1-f)*np.abs(cp[:,i].mean()-cd[:,i]))
    Y = np.log(pow(V,1-f)*np.abs(cp[:,i].mean()-cd[:,i])).reshape(-1,1)
    lr = LinearRegression()  # create object for the class
    lr.fit(X, Y)  # perform linear 
    x_W = -lr.coef_*V.mean()
    predicted_cd = np.empty(241)
    predicted_cd[0] = cd[0,i]
    
    for t in range(1,241):
        predicted_cd[t] = cp[:,i].mean()-(cp[:,i].mean()-predicted_cd[0])*((V[0]/V[t])**(1-f))*np.exp(-x_W[0,0]/V.mean()*t)
       
    return np.sqrt(((df_cd.iloc[:,i]-[predicted_cd[x] for x in df_cd.index])**2).sum(axis = 0))

solutes = ["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"]

for pfile in map(os.path.basename,glob.glob("./patient_files/pig2/session1/P10.csv")):  
      
    t = 240 #min

    p = pfile.split(".")[0]
    print(p)
    
    df = pd.read_csv("./patient_files/pig2/session1/"+pfile,skiprows = range(0,16), delimiter = "," \
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
    df_V = pd.read_csv("./patient_files/pig2/session1/"+pfile,skiprows = range(0,45), delimiter = ",", \
                       encoding= 'unicode_escape')[["IP volume T=0 (mL)","IP volume T=240 (mL)"]].iloc[0] # IPV measured from haemoglobin
    df_V=df_V.astype(float)
    print(df_V)
    
    f_cp = interp1d(df_cp.index, df_cp, axis = 0)
    cp = f_cp(range(0,t+1))
    
    #Linear interpolation to find values of cp at all times
    
    f_cd = interp1d(df_cd.index, df_cd, axis = 0)
    cd = f_cd(range(0,t+1))

    #Linear interpolation to find values of V at all times
    f_V = interp1d([0,240], df_V)
    V = f_V(range(0,t+1))
    
    solutes = ["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"]
    
    "Optimisaiton routine"
    X = np.array(range(0,241)).reshape(-1,1)
    fsel = []
    for i, solute in enumerate(solutes):
        f = 0.5
        result = scipy.optimize.minimize(objective_fn, f, args = (i, cp, V, df_cd, cd),
                method='SLSQP', bounds = [(-10,10)],
                options = {"maxiter" : 1000, "disp": False})
        fsel.append(result['x'])
        
    "predicted_cd with selected f values"
    predicted_cd_1 = pd.DataFrame(columns= ["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"]) 
    predicted_cd_1.loc[0]=df_cd.loc[0] #the value of predicted cd0 comes from the initial guess
    predicted_cd_1 = objective(fsel, i, cp, V, df_cd, cd, predicted_cd_1)
    
    "predicted values with fixed f at 0.5 but MTAC calculated from all data points"
    predicted_cd_05 = pd.DataFrame(columns= ["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"])
    predicted_cd_05.loc[0]=df_cd.loc[0]
    predicted_cd_05 = objective([0.5]*6, i, cp, V, df_cd, cd, predicted_cd_05)
    
    "predicted values with fixed f at 0.5 but MTAC calculated from only first and last data point"    
    predicted_cd_3 = pd.DataFrame(columns= ["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"])
    predicted_cd_3.loc[0]=df_cd.loc[0]
    predicted_cd_3 = objective3(cp, V, df_cd, cd, predicted_cd_3)
    
    "PLOT"
    fig, ax = plt.subplots(3,2, figsize = (12,18))
    t = 240
    #urea
    df_cd['Urea'].plot( ax = ax[0,0], label = 'data', style = '.')
    ax[0,0].plot(np.arange(t+1),predicted_cd_1['Urea'], label = 'predicted')
    ax[0,0].plot(np.arange(t+1),predicted_cd_05['Urea'], label = 'predicted05')
    ax[0,0].plot(np.arange(t+1),predicted_cd_3['Urea'], label = 'predicted3')
    ax[0,0].text(0.6, 0.1, f'f = {fsel[0]} ', transform=ax[0,0].transAxes)
    ax[0,0].set_title("Urea")
    
    #creatinine
    df_cd['Creatinine'].plot( ax = ax[0,1], style = '.')
    ax[0,1].plot(np.arange(t+1),predicted_cd_1['Creatinine'])
    ax[0,1].plot(np.arange(t+1),predicted_cd_05['Creatinine'])
    ax[0,1].plot(np.arange(t+1),predicted_cd_3['Creatinine'])
    ax[0,1].text(0.6, 0.1, f'f = {fsel[1]} ', transform=ax[0,1].transAxes)
    ax[0,1].set_title("Creatinine")
    
    #Sodium
    df_cd['Sodium'].plot( ax = ax[1,0],  style = '.')
    ax[1,0].plot(np.arange(t+1),predicted_cd_1['Sodium'] )
    ax[1,0].plot(np.arange(t+1),predicted_cd_05['Sodium'] )
    ax[1,0].plot(np.arange(t+1),predicted_cd_3['Sodium'] )
    ax[1,0].text(0.6, 0.5, f'f = {fsel[2]}', transform=ax[1,0].transAxes)
    ax[1,0].set_title("Sodium")
    
    #Phosphate
    df_cd['Phosphate'].plot( ax = ax[1,1], style = '.')
    ax[1,1].plot(np.arange(t+1),predicted_cd_1['Phosphate'] )
    ax[1,1].plot(np.arange(t+1),predicted_cd_05['Phosphate'] )
    ax[1,1].plot(np.arange(t+1),predicted_cd_3['Phosphate'] )
    ax[1,1].text(0.6, 0.1, f'f = {fsel[3]} ', transform=ax[1,1].transAxes)
    ax[1,1].set_title("Phosphate")
    
    #Glucose
    df_cd['Glucose'].plot( ax = ax[2,0], style = '.')
    ax[2,0].plot(np.arange(t+1),predicted_cd_1['Glucose'])
    ax[2,0].plot(np.arange(t+1),predicted_cd_05['Glucose'])
    ax[2,0].plot(np.arange(t+1),predicted_cd_3['Glucose'])
    ax[2,0].text(0.6, 0.5, f'f = {fsel[4]}', transform=ax[2,0].transAxes)
    ax[2,0].set_title("Glucose")
    
    #Potassium
    df_cd['Potassium'].plot( ax = ax[2,1], style = '.')
    ax[2,1].plot(np.arange(t+1),predicted_cd_1['Potassium'])
    ax[2,1].plot(np.arange(t+1),predicted_cd_05['Potassium'])
    ax[2,1].plot(np.arange(t+1),predicted_cd_3['Potassium'])
    ax[2,1].text(0.6, 0.1, f'f= {fsel[5]} ', transform=ax[2,1].transAxes)
    ax[2,1].set_title("Potassium")
    
    ax[0,0].legend()
    fig.supxlabel("time, min")
    fig.supylabel("Dialysate concentration, mmol")
    plt.show()

