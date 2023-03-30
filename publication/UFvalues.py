# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:30:03 2022
***model 7 specific***
This has some new parameters extracted directly from the patient files, such as L, delP
Also V is in ml here - needed for model 7
@author: P70073624
"""

import pandas as pd
from scipy.interpolate import interp1d
import numpy as np

def input_values(pfile):
    
    solutes = ["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"]
    t = 240 #min
    #p = input("Enter patient's file number")
    #pfile = p +".csv"
    #pfile = "2019415_1.csv"
    p = pfile.split(".")[0]
    # print(p)
    df = pd.read_csv(pfile,skiprows = range(0,16), delimiter = "," \
                     , encoding= 'unicode_escape')
    df = df.replace('#DIV/0!', np.nan)
    df = df.replace('#VALUE!', np.nan)
    # print(df.head())
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
    
    '''colloid osmotic pressure'''
    albumin = df['Albumin'].iloc[1:4].copy().astype('float64').set_axis(index).interpolate(method = 'index', limit_direction = "both")
    TP = df['Total protein'].iloc[1:4].copy().astype('float64').set_axis(index).interpolate(method = 'index', limit_direction = "both") #total protein
    TPm = TP.mean()/10 #to convert into g/dL
    alb_fr = albumin.mean()/TP.mean()
    delP0 = alb_fr*(2.8* TPm +0.18*TPm**2 + 0.012*TPm**3) + (1-alb_fr)*(0.9* TPm +0.12*TPm**2 + 0.004*TPm**3) 
    #corrected equation for Landis Pappenheimer, Nitta et al. doi:10.1620/tjem.135.43
    if np.isnan(delP0):
        delP0 = 8

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
    # print(df_cd)

    '''dialysate volume'''
    df_V = pd.read_csv(pfile,skiprows = range(0,45), delimiter = ",", \
                       encoding= 'unicode_escape')[["IP volume T=0 (mL)","IP volume T=240 (mL)"]].iloc[0] # IPV measured from haemoglobin
    print(df_V)
    
    #Linear interpolation to find values of cp at all times
    
    f_cp = interp1d(df_cp.index, df_cp, axis = 0)
    interpolated_cp = f_cp(range(0,t+1))
    
    #Linear interpolation to find values of cp at all times
    
    f_cd = interp1d(df_cd.index, df_cd, axis = 0)
    interpolated_cd = f_cd(range(0,t+1))

    #Linear interpolation to find values of V at all times
    f_V = interp1d([0,240], df_V)
    interpolated_V = f_V(range(0,t+1))

    #predicted_cd[0] = df_cd["cd"][0]
    
    Cp = interpolated_cp
    cd = interpolated_cd
    V = interpolated_V
    
    cd0_m, MTAC_m, fct_m, SiCo_m, QL_m, AIC =[np.empty(6, dtype = object) for _ in range(6)]
    
    #cols = ["model", "cd0", "MTAC","fct","SiCo","QL", "Guess no.", "evaluation", "AIC_score", "Initital guess"]
    
    predicted_cd = pd.DataFrame(columns= ["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"])
    
    predicted_cd.loc[0]=df_cd.loc[0] #the value of predicted cd0 comes from the initial guess
    
    # Residual volume ... total protein
    Vr = float(pd.read_csv(pfile,skiprows = range(0,45), delimiter = ",", \
                       encoding= 'unicode_escape')["RV before SPA (mL)"].iloc[1]) #ml # Changing back to normal values
    # if total protein RV cant be determined, then use creatinine then albumin
    if np.isnan(Vr):
        Vr = float(pd.read_csv(pfile,skiprows = range(0,45), delimiter = ",", \
                           encoding= 'unicode_escape')["RV before SPA (mL)"].iloc[0]) 
    if np.isnan(Vr):
        Vr = float(pd.read_csv(pfile,skiprows = range(0,45), delimiter = ",", \
                           encoding= 'unicode_escape')["RV before SPA (mL)"].iloc[2])         
    
    # fill volume
    V_fill = float(pd.read_csv(pfile,skiprows = range(0,39), delimiter = ",", \
                       encoding= 'unicode_escape')["Volume instilled (mL)"].iloc[1]) #ml Changing back to normal values
    if np.isnan(V_fill):
        V_fill = 2000 # mL. in case there are no measurement values
    df_drain = pd.DataFrame(columns = ["Creatinine", "Total protein", "Albumin"],dtype = float)
    df_drain = df[["Creatinine", "Total protein", "Albumin"]].iloc[8:11].copy()  #dialysate concentration
    for column in df_drain:
        df_drain[column] = df_drain[column].astype('float64') 
    df_drain.loc[:, "Creatinine"]  *= 0.001

    for col in df_drain.columns:   
        if np.isnan(Vr):
            Vr = V_fill*df_drain.loc[10,col]/(df_drain.loc[10,col]-df_drain.loc[8,col])

    if np.isnan(Vr):
        Vr = 200 # mL. in case there are no measurement values
    # if df.loc[47, 'Urea'] == '#DIV/0!' or df.loc[47, 'Urea'] == 'NaN':
    #     L = 0.7
    # else:
    # col_inx = list(df.columns ).index(df.columns[(df.values=='ELAR (mL/min)').any(0)])
    # L = float(df.iloc[(df.iloc[:,col_inx] == 'ELAR (mL/min)').idxmax(),col_inx+4])
    # if np.isnan(L):
    L = 0.7 #Morphometry and Lymph Dynamics of Swine Thoracic Duct
    
    df = df.replace('#DIV/0!', np.nan)
    # UF = df.loc[47:53, 'Sodium'].astype('float64')
    # index = pd.Series([10,20,30,60,120,180, 240])
    # UF = UF.set_axis(index)
    predicted_V = np.zeros(t+1)
    predicted_V[0] = V[0]
    
    return  Cp, V, cd,  L, delP0, V_fill, Vr, predicted_V,predicted_cd, df_cd

