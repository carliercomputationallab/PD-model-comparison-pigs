# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 11:03:52 2022
Figure 3
@author: P70073624
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

font = {'family':'sans','weight':'normal',
        'size'   : 16}

plt.rc('font', **font)
xls = pd.ExcelFile('results_RMSEtotal-true.xlsx')

comp = pd.read_excel('computationaltime_1patient_10startingpoints.xlsx').set_index('model')

#for 7: 4 train test split
df1 = pd.read_excel(xls, 'Sheet1', usecols='A:E', names = ['model', 'dataset', '1', '2', '3'])
df1= df1.replace(np.inf, np.nan)

df1.iloc[::3,2:5] = df1.iloc[::3,2:5]/4/8 #average over number of datasets, i.e.
# test set for 7-4 = 4 in the first session and then over the number of timesteps
df1.iloc[1::3,2:5] = df1.iloc[1::3,2:5]/5/8
df1.iloc[2::3,2:5] = df1.iloc[2::3,2:5]/7/8
df2 = df1.iloc[:,2:5]

df2.where(df2<6000, 6000, inplace = True)
# df2.where(df2<6000, np.nan, inplace = True) #ignoring the higher value does 
# not penalise the models that do not provide results for all iterations
df1.iloc[:,2:5] = df2
df1["mean"] = df1[['1', '2', '3']].mean(axis =1)
df1["std"] = df1[['1', '2', '3']].std(axis =1)

#for 6:5 train test split
df3 = pd.read_excel(xls, 'Sheet2', usecols='A:E', names = ['model', 'dataset', '1', '2', '3'])
df3= df3.replace(np.inf, np.nan)

df3.iloc[::3,2:5] = df3.iloc[::3,2:5]/5/8
df3.iloc[1::3,2:5] = df3.iloc[1::3,2:5]/5/8
df3.iloc[2::3,2:5] = df3.iloc[2::3,2:5]/6/8
df4 = df3.iloc[:,2:5]

df4.where(df4<6000, 6000, inplace = True)
# df2.where(df2<6000, np.nan, inplace = True) #ignoring the higher value does 
# not penalise the models that do not provide results for all iterations
df3.iloc[:,2:5] = df4
df3["mean"] = df3[['1', '2', '3']].mean(axis =1)
df3["std"] = df3[['1', '2', '3']].std(axis =1)


#%%
fig, axes = plt.subplots(2,3,figsize = (20,12))
for i,ax in enumerate(axes.flat):
    # ax.set_ylim(-10,300)
    ax.grid(True, ls = '--', lw = 0.3)
    ax.set_yscale('log')


keys = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9']
row = 0
for key in keys:
    df1.iloc[row:(row+3), 0] = key
    df3.iloc[row:(row+3), 0] = key
    row = row + 3

df_same = df1.loc[df1['dataset'] == 'same session']
df_other = df1.loc[df1['dataset'] == 'other session']
df_training = df1.loc[df1['dataset'] == 'training set']

axes[0,0].errorbar(df_training['model'], df_training['mean'], yerr = df_training['std'], ls = '', marker = 's', capsize = 2.0, color = 'k')
axes[0,1].errorbar(df_same['model'], df_same['mean'], yerr = df_same['std'], ls = '', marker = 's', capsize = 2.0, color = 'k')
axes[0,2].errorbar(df_other['model'], df_other['mean'], yerr = df_other['std'], ls = '', marker = 's', capsize = 2.0, color = 'k')

ax2 = axes[0,2].twinx()
ax2.errorbar(df_training['model'], comp['mean'], comp['sd'], ls = '', marker = 's', capsize = 2.0, color = 'red')
ax2.set_ylabel('Computational time, s', color = 'red')
ax2.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
#------------------------------------------#
#    OUTPUT THE RESULTS TO THE CONSOLE     #
#------------------------------------------#
for x,y in zip(df_same['mean'], df_same['std']):
    print(f'{x:.3f} \u00B1 {y:.3f}')
print("---------------------------")       
for x,y in zip(df_other['mean'], df_other['std']):
    print(f'{x:.3f} \u00B1 {y:.3f}')
print("---------------------------")       
for x,y in zip(df_training['mean'], df_training['std']):
    print(f'{x:.3f} \u00B1 {y:.3f}')
print("---------------------------")      
    
#------------------------------------------#
#                OTHER RATIO               #
#------------------------------------------#
df_same = df3.loc[df3['dataset'] == 'same session']
df_other = df3.loc[df3['dataset'] == 'other session']
df_training = df3.loc[df3['dataset'] == 'training set']

axes[1,0].errorbar(df_training['model'], df_training['mean'], yerr = df_training['std'], ls = '', marker = 'o', capsize = 2.0, color='darkorange')
axes[1,1].errorbar(df_same['model'], df_same['mean'], yerr = df_same['std'], ls = '', marker = 'o', capsize = 2.0, color='darkorange')
axes[1,2].errorbar(df_other['model'], df_other['mean'], yerr = df_other['std'], ls = '', marker = 'o', capsize = 2.0, color='darkorange')


for x,y in zip(df_same['mean'], df_same['std']):
    print(f'{x:.3f} \u00B1 {y:.3f}')
print("---------------------------")       
for x,y in zip(df_other['mean'], df_other['std']):
    print(f'{x:.3f} \u00B1 {y:.3f}')
print("---------------------------")       
for x,y in zip(df_training['mean'], df_training['std']):
    print(f'{x:.3f} \u00B1 {y:.3f}')

#------------------------------------------#
#               PLOT DETAILS               #
#------------------------------------------#
axes[0,0].set_title("Training set")
axes[0,1].set_title("Test set-first session")
axes[0,2].set_title("Test set-second session")


axes[0,0].set_ylabel('Total RMSE')
axes[1,0].set_ylabel('Total RMSE')

blue_circle = mlines.Line2D([0], [0], marker='o', color='w', label='6:5 split',
                          markerfacecolor='darkorange', markersize=10)
axes[1,2].legend(handles=[blue_circle])

black_square = mlines.Line2D([0], [0], marker='s', color='w', label='7:4 split',
                          markerfacecolor='k', markersize=10)
axes[0,2].legend(handles=[black_square])


for i, label in enumerate(('A', 'B', 'C', 'D', 'E', 'F')):
    axes.flat[i].text(0.05, 0.9, label, transform=axes.flat[i].transAxes,
            fontsize=16, fontweight='bold', va='top')

# ax[1,0].set_ylabel('----8/3 ratio----')

fig.supxlabel('Model')

keys = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9']
axes[0,0].set_xticks(range(0,9), keys)
axes[0,1].set_xticks(range(0,9), keys)
axes[0,2].set_xticks(range(0,9), keys)

plt.subplots_adjust(top=0.9,
                    bottom=0.145,
                    left=0.085,
                    right=0.96,
                    hspace=0.2,
                    wspace=0.2)

fig.tight_layout()
