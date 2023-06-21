# import cyipopt 
import numpy as np
import os
import glob
# import numdifftools.nd_statsmodels as nd   
import matplotlib.pyplot as plt 
import scipy
from scipy.optimize import minimize
import pandas as pd
from scipy.interpolate import interp1d
import random
# import statistics
import csv
# from csv import writer
from itertools import count


def objective(mtac, predicted_cd, Cp, L, V, df_cd, Vr, V_fill, x0, x1):
    '''The objective function needed to be minimised'''
    
    
    t = 240
    
    

    predicted_cd = rk(t, mtac, predicted_cd, Cp, L, V, df_cd, Vr, V_fill, x0, x1)

    
    return predicted_cd

def rk(t, mtac, predicted_cd, Cp, L, V, df_cd, Vr, V_fill, x0, x1):
    
    for timestep in range(0,t): 
        
        cd = predicted_cd.loc[timestep]
        
        PS_s = mtac[0:6] * 0.998 * abya0_s* (1 + x0* np.exp(-timestep/x1)) #fraction small pore surface area - Rippe, A THREE-PORE MODEL OF PERITONEAL TRANSPORT table 1
        # print(PS_s)
        #The current albumin value is from Joost.
        #MTAC for large pores
        PS_l = mtac[0:6] * 0.002 *abya0_l* (1 + x0* np.exp(-timestep/x1))# Ref: two pore model-oberg,rippe, table 1, A0L/A0 value
        
        "Apply Runge Kutta Formulas to find next value of y"
        k1, v1 = comdxdt(cd, timestep, mtac,  predicted_cd, Cp, L, V, df_cd, Vr, V_fill, PS_s, PS_l)
        k2, v2 = comdxdt(cd + 0.5  *k1, timestep, mtac,  predicted_cd, Cp, L, V, df_cd, Vr, V_fill, PS_s, PS_l)
        k3, v3 = comdxdt(cd + 0.5  *k2, timestep, mtac,  predicted_cd, Cp, L, V, df_cd, Vr, V_fill, PS_s, PS_l)
        k4, v4 = comdxdt(cd + k3, timestep, mtac,  predicted_cd, Cp, L, V, df_cd, Vr, V_fill, PS_s, PS_l)
        
        # Update next value of y
        cd = cd + (1 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
        
        #print(UF)
        predicted_cd.loc[timestep+1] = cd
        
        
    return predicted_cd


# the differential equations
def comdxdt(cd, t, x, predicted_cd, Cp, L, V, df_cd, Vr, V_fill, PS_s, PS_l):
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
    
    
    af = 16.18 * (1 - np.exp (-0.00077*V[t]))/13.3187
    
    delP = delP0 - ((V[t] - (V_fill+Vr))/490)#consider getting Vfill and Vr from the data file also
    #peritoneal concentration gradient
   
    
    pr = [phi[i] *RT * (Cp[t][i]-cd[i]) for i in range(len(solutes))]
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
    dxdt = ((J_sS + J_sL)/V[t]-np.array(cd)*(J_vC + J_vS + J_vL-L)/V[t]).ravel()
    # print(dxdt)
    return (dxdt, J_vC+J_vS+J_vL)


#%%

solutes = ["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"]
   
pfile = './patient_files/pig2/session1/P10.csv'
      
t = 240 #min

p = pfile.split("/")[4]
print(p)

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

#########################################
#           GARRED & WANIEWSKI          #
#########################################
MTAC_G = df_V.mean()/t*np.log(
    (df_V.loc["IP volume T=0 (mL)"]*(df_cp.mean()-df_cd.loc[0]))/
    (df_V.loc["IP volume T=240 (mL)"]*(df_cp.mean()-df_cd.loc[240])))


MTAC_W = df_V.mean()/t*np.log(
    (pow(df_V.loc["IP volume T=0 (mL)"],(1/2))*(df_cp.mean()-df_cd.loc[0]))/
    (pow(df_V.loc["IP volume T=240 (mL)"],(1/2))*(df_cp.mean()-df_cd.loc[240])))
#%%
predicted_cd_G = pd.DataFrame(columns= ["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"])
predicted_cd_G.loc[0]=df_cd.loc[0]

predicted_cd_W = pd.DataFrame(columns= ["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"])
predicted_cd_W.loc[0]=df_cd.loc[0]

f_V = interp1d([0,240], df_V)
interpolated_V = f_V(range(0,t+1))
V = interpolated_V/1000
for t in range(1,241):
    predicted_cd_G.loc[t] = df_cp.mean(axis=0)-(1/V[t])*V[0]*(df_cp.mean(axis=0)-predicted_cd_G.loc[0])*np.exp(-MTAC_G*t/V.mean()/1000)
    predicted_cd_W.loc[t] = df_cp.mean(axis=0)-(df_cp.mean(axis=0)-predicted_cd_W.loc[0])*((V[0]/V[t])**(1-0.5))*np.exp(-MTAC_W/V.mean()*t/1000)

#########################################
#                OBERG                  #
#########################################    
oberg = pd.read_excel('fitted_X.xlsx').set_index('patient')
MTAC_O = oberg.loc[p].iloc[0:6]
predicted_cd_O = pd.DataFrame(columns= ["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"])
predicted_cd_O.loc[0]=df_cd.loc[0]
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

f_cp = interp1d(df_cp.index, df_cp, axis = 0)
interpolated_cp = f_cp(range(0,t+1))
Cp = interpolated_cp
V = interpolated_V
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
predicted_cd_O = objective(MTAC_O, predicted_cd_O, Cp, L, V, df_cd, Vr, V_fill, 0, 1)



#########################################
#                   PLOT                #
#########################################
    
fig, ax = plt.subplots(3,2, figsize = (12,18))
ax[0,0].scatter(df_cd.index,df_cd.iloc[:,0], label = 'Data', c = 'k') 
ax[0,0].plot(range(241),predicted_cd_G['Urea'], label = 'Garred', lw = 3)  
ax[0,0].plot(range(241),predicted_cd_W['Urea'], label = 'Waniewski', ls = '--', lw = 3) 
ax[0,0].plot(range(241),predicted_cd_O['Urea'], label = 'TPM',ls = '-.', lw = 3) 

ax[0,0].set_title('Urea')
ax[0,1].scatter(df_cd.index,df_cd.iloc[:,1], label = 'Data', c = 'k') 
ax[0,1].plot(range(241),predicted_cd_G['Creatinine'], label = 'fitted', lw = 3) 
ax[0,1].plot(range(241),predicted_cd_W['Creatinine'], label = 'fitted', ls = '--', lw = 3) 
ax[0,1].plot(range(241),predicted_cd_O['Creatinine'], label = 'fitted',ls = '-.', lw = 3)

ax[0,1].set_title('Creatinine')
ax[1,0].scatter(df_cd.index,df_cd.iloc[:,2], label = 'Data', c = 'k') 
ax[1,0].plot(range(241),predicted_cd_G['Sodium'], label = 'fitted', lw = 3) 
ax[1,0].plot(range(241),predicted_cd_W['Sodium'], label = 'fitted', ls = '--', lw = 3)
ax[1,0].plot(range(241),predicted_cd_O['Sodium'], label = 'fitted',ls = '-.', lw = 3)  

ax[1,0].set_title('Sodium')
ax[1,1].scatter(df_cd.index,df_cd.iloc[:,3], label = 'Data', c = 'k') 
ax[1,1].plot(range(241),predicted_cd_G['Phosphate'], label = 'fitted', lw = 3)  
ax[1,1].plot(range(241),predicted_cd_W['Phosphate'], label = 'fitted', ls = '--', lw = 3)
ax[1,1].plot(range(241),predicted_cd_O['Phosphate'], label = 'fitted',ls = '-.', lw = 3) 

ax[1,1].set_title('Phosphate')
ax[2,0].scatter(df_cd.index,df_cd.iloc[:,4], label = 'Data', c = 'k') 
ax[2,0].plot(range(241),predicted_cd_G['Glucose'], label = 'fitted', lw = 3)
ax[2,0].plot(range(241),predicted_cd_W['Glucose'], label = 'fitted', ls = '--', lw = 3)
ax[2,0].plot(range(241),predicted_cd_O['Glucose'], label = 'fitted',ls = '-.', lw = 3)

ax[2,0].set_title('Glucose')
ax[2,1].scatter(df_cd.index,df_cd.iloc[:,5], label = 'Data', c = 'k') 
ax[2,1].plot(range(241),predicted_cd_G['Potassium'], label = 'Garred', lw = 3)
ax[2,1].plot(range(241),predicted_cd_W['Potassium'], label = 'Waniewski', ls = '--', lw = 3)
ax[2,1].plot(range(241),predicted_cd_O['Potassium'], label = 'TPM',ls = '-.', lw = 3)

ax[2,1].set_title('Potassium')

L = 0.7
LS = 0.045
alpha = [0.052, 0.900, 0.048]
predicted_cd_O2 = objective(MTAC_O, predicted_cd_O, Cp, L, V, df_cd, Vr, V_fill, 0, 1)
ax[0,0].plot(range(241),predicted_cd_O2['Urea'], label = 'TPM-N') 
ax[0,1].plot(range(241),predicted_cd_O2['Creatinine'], label = 'fitted')
ax[1,0].plot(range(241),predicted_cd_O2['Sodium'], label = 'fitted') 
ax[1,1].plot(range(241),predicted_cd_O2['Phosphate'], label = 'fitted') 
ax[2,0].plot(range(241),predicted_cd_O2['Glucose'], label = 'fitted')
ax[2,1].plot(range(241),predicted_cd_O2['Potassium'], label = 'TPM-N')


predicted_cd_O3 = objective(MTAC_O, predicted_cd_O, Cp, L, V, df_cd, Vr, V_fill, 10.0084, 1.7744)
ax[0,0].plot(range(241),predicted_cd_O3['Urea'], label = 'TPM-MTAC') 
ax[0,1].plot(range(241),predicted_cd_O3['Creatinine'], label = 'fitted')
ax[1,0].plot(range(241),predicted_cd_O3['Sodium'], label = 'fitted') 
ax[1,1].plot(range(241),predicted_cd_O3['Phosphate'], label = 'fitted') 
ax[2,0].plot(range(241),predicted_cd_O3['Glucose'], label = 'fitted')
ax[2,1].plot(range(241),predicted_cd_O3['Potassium'], label = 'TPM-MTAC')

ax[1,0].set_ylim(ymax = 131)

fig.supylabel('cd, mmol/L')
fig.supxlabel('time, min')
fig.tight_layout()
plt.legend()

plt.savefig('Figure_4.eps', dpi = 600)