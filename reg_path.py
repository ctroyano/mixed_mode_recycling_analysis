import pandas as pd
import statsmodels.formula.api as smf
import IPython
import numpy as np
from model_loss import model_loss_func
from param_num import param_num_func
import matplotlib.pyplot as plt

######################################################################
#DESCRIPTION
#this file makes a lasso regularization path for the OLS model used to calculate coating separate effects on samples 51, 51 coat sep, 
#52, 52 coat sep, 68, and 68 coat sep 
######################################################################

mod_loss_toughness = []
mod_loss_elongation = []
mod_loss_youngs = []
mod_loss_max_strength = []
param_no_toughness = []
param_no_elongation = []
param_no_youngs = []
param_no_max_strength = []

alphas = np.logspace(4, 5, 100)
alphas_elongation = np.logspace(-2, 3, 1000)

for elem in alphas:
    loss_toughness = model_loss_func(elem, 'toughness')
    loss_youngs = model_loss_func(elem, 'young')
    loss_max_strength = model_loss_func(elem, 'strength')

    param_toughness = param_num_func(elem, 'toughness')
    param_youngs = param_num_func(elem, 'young')
    param_max_strength = param_num_func(elem, 'strength')

    mod_loss_toughness.append(loss_toughness)
    mod_loss_youngs.append(loss_youngs)
    mod_loss_max_strength.append(loss_max_strength)

    param_no_toughness.append(param_toughness)
    param_no_youngs.append(param_youngs)
    param_no_max_strength.append(param_max_strength)

for elem in alphas_elongation:
    loss_elongation = model_loss_func(elem, 'elongation')
    param_elongation = param_num_func(elem, 'elongation')
    mod_loss_elongation.append(loss_elongation)
    param_no_elongation.append(param_elongation)

#for


#TOUGHNESS GRAPHS
ax1 = plt.subplot(211)
#fig1, ax1 = plt.subplots()
plt.plot(np.log10(alphas), param_no_toughness, '-')
plt.xlabel('log(Alpha)')
plt.ylabel('number of parameters')

#plt.show()

#plotting model loss v. alphas, model loss in MPa
mod_loss_toughness = np.array(mod_loss_toughness)
mod_loss_toughness = mod_loss_toughness/10**6

ax3 = plt.subplot(212)
#fig3, ax3 = plt.subplots()
plt.plot(np.log10(alphas), mod_loss_toughness, '-')
plt.ylabel('model loss (MPa)')
plt.xlabel('log(Alpha)')
plt.show()

#ELONGATION
#alphas needs to be much smaller for elongation. on the order of 0 to 100. 
ax4 = plt.subplot(211)
#fig4, ax4 = plt.subplots()
plt.plot(np.log10(alphas_elongation), param_no_elongation, '-')
plt.xlabel('log(Alpha)')
plt.ylabel('number of parameters')


ax5 = plt.subplot(212)
#fig_new, ax_new = plt.subplots()
plt.plot(np.log10(alphas_elongation), mod_loss_elongation, '-')
plt.xlabel('log(Alpha)')
plt.ylabel('model loss (%)')
plt.show()


#MAX STRENGTH
ax6 = plt.subplot(211)
#fig6, ax6 = plt.subplots()
plt.plot(np.log10(alphas), param_no_max_strength, '-')
plt.xlabel('log(Alpha)')
plt.ylabel('number of parameters')


#plotting model loss v. alphas, model loss in MPa

mod_loss_max_strength = np.array(mod_loss_max_strength)
mod_loss_max_strength = mod_loss_max_strength/10**6

ax7 = plt.subplot(212)
plt.plot(np.log10(alphas), mod_loss_max_strength, '-')
plt.ylabel('model loss (MPa)')
plt.xlabel('log(Alpha)')
plt.show()

#YOUNGS MODULUS
ax8 = plt.subplot(211)
#fig6, ax6 = plt.subplots()
plt.plot(np.log10(alphas), param_no_youngs, '-')
plt.xlabel('log(Alpha)')
plt.ylabel('number of parameters')


#plotting model loss v. alphas, model loss in MPa

mod_loss_youngs = np.array(mod_loss_youngs)
mod_loss_youngs = mod_loss_youngs/10**6

ax9 = plt.subplot(212)
plt.plot(np.log10(alphas), mod_loss_max_strength, '-')
plt.ylabel('model loss (MPa)')
plt.xlabel('log(Alpha)')
plt.title('modulus')
plt.show()

IPython.embed()