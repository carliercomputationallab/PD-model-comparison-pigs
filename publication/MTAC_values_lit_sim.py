# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:45:45 2023
plot oberg fitted parameters versus literature
@author: P70073624
"""
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

font = {'size'   : 18}

matplotlib.rc('font', **font)
plt.rcParams["font.family"] = "Times New Roman"

df = pd.read_excel('MTAC_values_lit_sim.xlsx')

solutes = ['Urea', 'Creatinine', 'Phosphate', 'Potassium']

y_SPA_peritonitis_no_avg = df['SPA'].iloc[0::2]
y_SPA_peritonitis_no_sd = df['SD-SPA'].iloc[0::2]

y_SPA_peritonitis_y_avg = df['SPA'].iloc[1::2]
y_SPA_peritonitis_y_sd = df['SD-SPA'].iloc[1::2]

y_Oberg_avg = df['Oberg (Pigs) - I fitted'].iloc[0::2]
y_Oberg_sd = df['oberg-SD'].iloc[0::2]

y_dV_median = df['deVries (pigs) - uncorrected for BSA'].iloc[0::2]
ymin = df['min'].iloc[0::2]
ymax = df['max'].iloc[0::2]
y_dV_sd = df['SD-devries'].iloc[0::2]

fig = plt.figure(figsize = (6,6))

offset = lambda p: transforms.ScaledTranslation(p/72.,0, plt.gcf().dpi_scale_trans)
trans = plt.gca().transData

plt.errorbar(solutes, y_SPA_peritonitis_no_avg, y_SPA_peritonitis_no_sd, c = '#007a60', capsize = 2.0, marker = 'd', ls = '', label = 'van Gelder et al.',  transform=trans+offset(-5))

# plt.errorbar(solutes, y_SPA_peritonitis_y_avg, y_SPA_peritonitis_y_sd, c = '#f9627d', capsize = 2.0, marker = '^', ls = '', label = 'VG_yes')



plt.errorbar(solutes, y_dV_median, [ymin, ymax], c = '#9c9c9c', capsize = 2.0, marker = 'o', ls = '')

plt.errorbar(solutes, y_dV_median, y_dV_sd, c = '#c73905', capsize = 2.0, marker = 'o', ls = '', label = 'de Vries et al.')

plt.errorbar(solutes, y_Oberg_avg, y_Oberg_sd, c = 'orange', capsize = 2.0, marker = 's', ls = '', label = 'TPM',  transform=trans+offset(5))

plt.ylabel('MTAC, ml/min')
plt.legend()