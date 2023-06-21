# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 13:27:19 2023
shortened to start from old time

figure 4
@author: P70073624
"""


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


font = {'family':'sans','weight':'normal',
        'size'   : 16}

plt.rc('font', **font)
font = {'size'   : 18}


plt.rcParams["font.family"] = "Times New Roman"

sse_model1 = pd.read_csv('sse_m1.csv', index_col = 0)
sse_model2 = pd.read_csv('sse_m2.csv', index_col = 0)
sse_model3 = pd.read_csv('sse_m3.csv', index_col = 0)
sse_model4 = pd.read_csv('sse_m4.csv', index_col = 0)
sse_model5 = pd.read_csv('sse_m5.csv', index_col = 0)
sse_model6 = pd.read_csv('sse_m6.csv', index_col = 0)
sse_model7 = pd.read_csv('sse_m7.csv', index_col = 0)
sse_model8 = pd.read_csv('sse_m8.csv', index_col = 0)
sse_model9 = pd.read_csv('sse_m9.csv', index_col = 0)

sse = [sse_model1, sse_model2, sse_model3, sse_model4, sse_model5, sse_model6, sse_model7, sse_model8, sse_model9 ]

    
solutes = ["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"]
keys = ['Test set-same session','SD1', 'Test set-other session','SD2', 'Training set', 'SD3']
df_list = []
for i,m in enumerate(sse):
    m = m/8 #number of time points
    m = m.replace([np.inf, -np.inf], np.nan)
    m.where(m < 6000, 6000, inplace=True)
    m.columns = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    print(m)
    df_list.append( pd.concat([m[0].mean(axis = 1, skipna = True),m[0].std(axis = 1, skipna = True),
                        m[1].mean(axis = 1, skipna = True),m[1].std(axis = 1, skipna = True),
                        m[2].mean(axis = 1, skipna = True),m[2].std(axis = 1, skipna = True)], axis = 1, 
                       keys = keys))

    # print(m[0].mean(axis = 1, skipna = True))
    

keys = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9']
#%%
# for df in df_list:
#     df.where(df < 1000, np.nan, inplace=True)
urea = (pd.concat([df.loc['Urea'] for df in df_list], axis = 1, keys = keys)).T.reset_index()
crea = pd.concat([df.loc['Creatinine'] for df in df_list], axis = 1, keys = keys).T.reset_index()
sodium = pd.concat([df.loc['Sodium'] for df in df_list], axis = 1, keys = keys).T.reset_index()
phos = pd.concat([df.loc['Phosphate'] for df in df_list], axis = 1, keys = keys).T.reset_index()
glu = pd.concat([df.loc['Glucose'] for df in df_list], axis = 1, keys = keys).T.reset_index()
pot = pd.concat([df.loc['Potassium'] for df in df_list], axis = 1, keys = keys).T.reset_index()

solutes = ["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"]
fig, axes = plt.subplots(nrows=3,ncols=2,figsize=(12,36))
colors = sns.color_palette(palette = 'colorblind', n_colors = 9, as_cmap=True)
# using dictionary comprehension
# to convert lists to dictionary
res = {keys[i]: colors[i] for i in range(len(keys))}
for i,ax in enumerate(axes.flat):
    ax.set_yscale('log')
    ax.set_ylim(0.007,100)
    ax.grid(True, ls = '--', lw = 0.3)
    ax.set_title(solutes[i])
    
#test-set same session

X = np.array(range(len(urea['index'])))

axes[0,0].errorbar(x = X - 0.1, y = urea['Test set-same session'], yerr = urea['SD1'], ls = '',color = 'k', marker = 's', capsize = 2.0)
axes[0,1].errorbar(x = X - 0.1, y = crea['Test set-same session'], yerr = crea['SD1'], ls = '', color = 'k', marker = 's', capsize = 2.0)
axes[1,0].errorbar(x = X - 0.1, y = sodium['Test set-same session'], yerr = sodium['SD1'], ls = '', color = 'k', marker = 's', capsize = 2.0, label = 'Test-1st')
axes[1,1].errorbar(x = X - 0.1, y = phos['Test set-same session'], yerr = phos['SD1'], ls = '', color = 'k', marker = 's', capsize = 2.0)
axes[2,0].errorbar(x = X - 0.1, y = glu['Test set-same session'], yerr = glu['SD1'], ls = '', color = 'k', marker = 's', capsize = 2.0)
axes[2,1].errorbar(x = X - 0.1, y = pot['Test set-same session'], yerr = pot['SD1'], ls = '', color = 'k', marker = 's', capsize = 2.0, label = 'Test-1st')

#test-set same session

axes[0,0].errorbar(x = X, y = urea['Test set-other session'], yerr = urea['SD2'], ls = '',color = 'crimson',marker = 'o', capsize = 2.0)
axes[0,1].errorbar(x = X, y = crea['Test set-other session'], yerr = crea['SD2'], ls = '', color = 'crimson',marker = 'o', capsize = 2.0)
axes[1,0].errorbar(x = X, y = sodium['Test set-other session'], yerr = sodium['SD2'], ls = '', color = 'crimson',marker = 'o', capsize = 2.0, label = 'Test-2nd')
axes[1,1].errorbar(x = X, y = phos['Test set-other session'], yerr = phos['SD2'], ls = '', color = 'crimson',marker = 'o', capsize = 2.0)
axes[2,0].errorbar(x = X, y = glu['Test set-other session'], yerr = glu['SD2'], ls = '', color = 'crimson',marker = 'o', capsize = 2.0)
axes[2,1].errorbar(x = X, y = pot['Test set-other session'], yerr = pot['SD2'], ls = '', color = 'crimson',marker = 'o', capsize = 2.0, label = 'Test-2nd')

#test-set same session

axes[0,0].errorbar(x = X + 0.1, y = urea['Training set'], yerr = urea['SD3'], ls = '',color = 'limegreen',marker = '*', capsize = 2.0)
axes[0,1].errorbar(x = X + 0.1, y = crea['Training set'], yerr = crea['SD3'], ls = '', color = 'limegreen',marker = '*', capsize = 2.0)
axes[1,0].errorbar(x = X + 0.1, y = sodium['Training set'], yerr = sodium['SD3'], ls = '', color = 'limegreen',marker = '*', capsize = 2.0, label = 'Training')
axes[1,1].errorbar(x = X + 0.1, y = phos['Training set'], yerr = phos['SD3'], ls = '', color = 'limegreen',marker = '*', capsize = 2.0)
axes[2,0].errorbar(x = X + 0.1, y = glu['Training set'], yerr = glu['SD3'], ls = '', color = 'limegreen',marker = '*', capsize = 2.0)
axes[2,1].errorbar(x = X + 0.1, y = pot['Training set'], yerr = pot['SD3'], ls = '', color = 'limegreen',marker = '*', capsize = 2.0, label = 'Training')

axes[2,1].legend(loc = 'upper right', frameon = True)
for ax in axes.flat:
    ax.set_xticks(range(len(urea['index'])))
    ax.set_xticklabels(urea['index'])

plt.grid(True, ls = '--', lw = 0.3)
fig.supylabel('RMSE', fontsize = 18)
fig.supxlabel('Model', fontsize = 18)
plt.subplots_adjust(top=0.9,
                    bottom=0.1,
                    left=0.07,
                    right=0.945,
                    hspace=0.2,
                    wspace=0.2)
fig.tight_layout()
plt.savefig('Figure_3.eps',  dpi = 300)
plt.show()