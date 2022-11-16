import numpy as np
import pandas as pd 
import IPython
import statsmodels.formula.api as smf
import math

############
#takes alpha and string and returns a model loss for toughness, max strength, elongation at break, or young's modulus 
############

#do this for maximum strength, elongation at break, and modulus 

lasso_df = pd.read_excel('lasso_data_table.xlsx')
lasso_df = lasso_df.drop(["Unnamed: 0", "Modulus at 50% (Pa)", "Modulus at 100% (Pa)", "Modulus at 200% (Pa)"], axis=1)
lasso_df.columns = ["sample_name", "Youngs_mod_pa", "elongation_break", "toughness_pa", "max_strength_pa", "A", "B", "D"]

#coat sep = A 
#reinforcement = B
#double_layer = C


def model_loss_func(alpha_in, str):
    if str.find("toughness") !=-1:
        mod_lasso_toughness_pa = smf.ols('toughness_pa~C(A)*C(B)*C(D)', data=lasso_df)
        res_lasso_toughness_pa = mod_lasso_toughness_pa.fit()   
        res_lasso_toughness_pa_reg = mod_lasso_toughness_pa.fit_regularized(alpha=alpha_in, L1_wt = 1, start_params=res_lasso_toughness_pa.params)
        predicted_results = res_lasso_toughness_pa_reg.predict({"A": lasso_df["A"], "B": lasso_df["B"], "D": lasso_df["D"]})
        difference = lasso_df["toughness_pa"] - predicted_results
        squared_difference = difference**2
        squared_dif_mean = squared_difference.mean()
        model_loss = math.sqrt(squared_dif_mean) #in Pa
        return model_loss
    if str.find("elongation") !=-1:
        mod_lasso_elongation = smf.ols('elongation_break~C(A)*C(B)*C(D)', data=lasso_df)
        res_lasso_elongation = mod_lasso_elongation.fit()   
        res_lasso_elongation_reg = mod_lasso_elongation.fit_regularized(alpha=alpha_in, L1_wt = 1, start_params=res_lasso_elongation.params)
        predicted_results = res_lasso_elongation_reg.predict({"A": lasso_df["A"], "B": lasso_df["B"], "D": lasso_df["D"]})
        difference = lasso_df["elongation_break"] - predicted_results
        squared_difference = difference**2
        squared_dif_mean = squared_difference.mean()
        model_loss = math.sqrt(squared_dif_mean) 
        return model_loss
    if str.find("young"): 
        mod_lasso_youngs_pa = smf.ols('Youngs_mod_pa~C(A)*C(B)*C(D)', data=lasso_df)
        res_lasso_youngs_pa = mod_lasso_youngs_pa.fit()   
        res_lasso_youngs_pa_reg = mod_lasso_youngs_pa.fit_regularized(alpha=alpha_in, L1_wt = 1, start_params=res_lasso_youngs_pa.params)
        predicted_results = res_lasso_youngs_pa_reg.predict({"A": lasso_df["A"], "B": lasso_df["B"], "D": lasso_df["D"]})
        difference = lasso_df["Youngs_mod_pa"] - predicted_results
        squared_difference = difference**2
        squared_dif_mean = squared_difference.mean()
        model_loss = math.sqrt(squared_dif_mean) 
        return model_loss
    if str.find("strength"):
        mod_lasso_max_strength_pa = smf.ols('max_strength_pa~C(A)*C(B)*C(D)', data=lasso_df)
        res_lasso_max_strength_pa = mod_lasso_max_strength_pa.fit()   
        res_lasso_max_strength_pa_reg = mod_lasso_max_strength_pa.fit_regularized(alpha=alpha_in, L1_wt = 1, start_params=res_lasso_max_strength_pa.params)
        predicted_results = res_lasso_max_strength_pa_reg.predict({"A": lasso_df["A"], "B": lasso_df["B"], "D": lasso_df["D"]})
        difference = lasso_df["max_strength_pa"] - predicted_results
        squared_difference = difference**2
        squared_dif_mean = squared_difference.mean()
        model_loss = math.sqrt(squared_dif_mean) 
        return model_loss
    else: 
        print("This function looks for toughness, elongation, young, and strength as its string arguments.")





   
   
   
   
   
   
   
    #for i, elem in enumerate(lasso_df["toughness"], start=0):
     #   sum = params.loc[['Intercept']].values[0] + 
    
        




    