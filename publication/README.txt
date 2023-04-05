****directory****
patient files -> has all the data separated by pig and session
spider_data -> all fitted parameters for teh three iterations of each model 1- 9 and both data splits 7:4 and 6:5

****Python files****
model 1 - 7 -train parallel.py -> has parallel processing routine for all model
-> save df_OV for fitted X for the particular model

model 8, 9.py -> runs on one processor - calculates MATC and then predicts the dialysate concetration

# for model 1- 9 save residuals in a file results_RMSEtotal-true.xlsx

model 9.py - fitting -> instead of forcing f to be 0.5 we can also fit f

model7-sodiumfitting-alphaultrasmall.py -> to fit alpha_c by minimising sodium error per patient basis

model7 - sodium fitting.py -> fitting sodium MTAC on a per patient basis

model 7 -Vfitting -> fitting LpS b yfitting only volume for all patients

model7_MTAC_with_time -> to fit a time dependent MTAC for all solutes

values - > this is where we extract all plasma and dialysate concentration directly from the patient files

UF values -> *for model 7 only* In addition to all things in values, we also extract delP, Vres, L

totalerror_per_model_figure3 -> this is to calculate total error per model and for both split ratio

import all sse - this runs all sse_per_solute_model* files to collect the error for each solute separately. It takes MTAC directly from spider data folder
and collects the mean and SD of the solute specific error for all iterations and each dataset.

solute_specific_error - this is to be run after import all sse to plot figure 4

sse_per_solute_model* - this is what runs from within import all sse. it collects the fitted values from spiderdata folder and reruns the simulations using a mean value 
to calculate the error per solute.

model789-predicted-solute-concentration-figure5 - for the best fitted mdoels, this takes those fitted parameters and plots the predicted dialysate concentration


****Excel files****
fitted_X.xlsx -> fitted x values from model 7 to be used to fit alpha and LpS later.
fitted_LpS -> all the LpS values by fitting volume 
results_RMSEtotal-true -> all residuals saved from model 1- 9
computational_time -> computational time required to run P10 only for 10 iterations on the server