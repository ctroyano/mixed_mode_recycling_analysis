import numpy as np
import pandas as pd 
import IPython
import statsmodels.formula.api as smf
import math

lasso_df = pd.read_excel('lasso_data_table.xlsx')
lasso_df = lasso_df.drop(["Unnamed: 0", "Modulus at 50% (Pa)", "Modulus at 100% (Pa)", "Modulus at 200% (Pa)"], axis=1)
lasso_df.columns = ["sample_name", "Youngs_mod_pa", "elongation_break", "toughness_pa", "max_strength_pa", "A", "B", "D"]

    
def param_num_func(alpha_in, str):

    if str.find("toughness") !=-1:
        mod_lasso_toughness_pa = smf.ols('toughness_pa~C(A)*C(B)*C(D)', data=lasso_df)
        res_lasso_toughness_pa = mod_lasso_toughness_pa.fit()   
        res_lasso_toughness_pa_reg = mod_lasso_toughness_pa.fit_regularized(alpha=alpha_in, L1_wt = 1, start_params=res_lasso_toughness_pa.params)
        count = 0
        for elem in res_lasso_toughness_pa_reg.params: 
            if abs(elem) > 0.0001: 
             count +=1
        return count 
    if str.find("elongation") !=-1:
        mod_lasso_elongation = smf.ols('elongation_break~C(A)*C(B)*C(D)', data=lasso_df)
        res_lasso_elongation = mod_lasso_elongation.fit()   
        res_lasso_elongation_reg = mod_lasso_elongation.fit_regularized(alpha=alpha_in, L1_wt = 1, start_params=res_lasso_elongation.params)
        count = 0
        for elem in res_lasso_elongation_reg.params: 
            if abs(elem) > 0.0001: 
             count +=1
        return count 

    if str.find("young") !=-1:
        mod_lasso_youngs_pa = smf.ols('Youngs_mod_pa~C(A)*C(B)*C(D)', data=lasso_df)
        res_lasso_youngs_pa = mod_lasso_youngs_pa.fit()   
        res_lasso_youngs_pa_reg = mod_lasso_youngs_pa.fit_regularized(alpha=alpha_in, L1_wt = 1, start_params=res_lasso_youngs_pa.params)
        count = 0
        for elem in res_lasso_youngs_pa_reg.params: 
            if abs(elem) > 0.0001: 
             count +=1
        return count 

    if str.find("strength") !=-1:
        mod_lasso_max_strength_pa = smf.ols('max_strength_pa~C(A)*C(B)*C(D)', data=lasso_df)
        res_lasso_max_strength_pa = mod_lasso_max_strength_pa.fit()   
        res_lasso_max_strength_pa_reg = mod_lasso_max_strength_pa.fit_regularized(alpha=alpha_in, L1_wt = 1, start_params=res_lasso_max_strength_pa.params)
        count = 0
        for elem in res_lasso_max_strength_pa_reg.params: 
            if abs(elem) > 0.0001: 
             count +=1
        return count    
    else: 
        print("This function looks for toughness, elongation, young, and strength as its string arguments.")