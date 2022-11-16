import pandas as pd
import statsmodels.formula.api as smf
import IPython


df = pd.read_excel('python_tables.xlsx')
del df['Unnamed: 0']
df.drop(df.iloc[:, 5:8], inplace=True, axis=1)
df.columns = ['sample_name', 'young_pa', 'elongation', 'toughness_pa','max_strength_pa',  'reinforcement', 'precure', 'cure_time', 'sulfur', 'double_layer']

######################################################################
#Yates analysis on the entire sample spread 
#factorial design analysis on the factorial experimental design 
#OLS regression on coating separate controls 
######################################################################

######################################################################
#FACTORIAL DESIGN
######################################################################

#max strength
mod_max_strength_pa = smf.ols(formula = 'max_strength_pa ~ C(reinforcement)*C(precure)*C(cure_time)*C(sulfur)*C(double_layer)', data = df)
res_max_strength_pa = mod_max_strength_pa.fit()
df_max_strength = pd.concat((res_max_strength_pa.params, res_max_strength_pa.bse, res_max_strength_pa.tvalues, res_max_strength_pa.pvalues), axis=1)
df_max_strength.columns = ['parameter (Pa)', 'standard error of effect (Pa)', 't_value', 'p_value' ]
df_max_strength.to_excel('yates_statsmodels_max_strength.xlsx', sheet_name = 'Sheet1')

#toughness
mod_toughness_pa = smf.ols(formula = 'toughness_pa ~ C(reinforcement)*C(precure)*C(cure_time)*C(sulfur)*C(double_layer)', data = df)
res_toughness_pa = mod_toughness_pa.fit()
df_toughness = pd.concat((res_toughness_pa.params, res_toughness_pa.bse, res_toughness_pa.tvalues, res_toughness_pa.pvalues), axis=1)
df_toughness.columns = ['parameter (Pa)', 'standard error of effect (Pa)', 't_value', 'p_value' ]
df_toughness.to_excel('yates_statsmodels_toughness.xlsx', sheet_name = 'Sheet1')

#elongation at break 
mod_elongation = smf.ols(formula = 'elongation ~ C(reinforcement)*C(precure)*C(cure_time)*C(sulfur)*C(double_layer)', data = df)
res_elongation = mod_elongation.fit()
df_elongation = pd.concat((res_elongation.params, res_elongation.bse, res_elongation.tvalues, res_elongation.pvalues), axis=1)
df_elongation.columns = ['parameter (%)', 'standard error of effect (%)', 't_value', 'p_value' ]
df_elongation.to_excel('yates_statsmodels_elongation.xlsx', sheet_name = 'Sheet1' )

#young's modulus 
mod_young_pa = smf.ols(formula = 'young_pa ~ C(reinforcement)*C(precure)*C(cure_time)*C(sulfur)*C(double_layer)', data = df)
res_young_pa = mod_young_pa.fit()
df_youngs = pd.concat((res_young_pa.params, res_young_pa.bse, res_young_pa.tvalues, res_young_pa.pvalues), axis=1)
df_youngs.columns = ['parameter (Pa)', 'standard error of effect (Pa)', 't_value', 'p_value' ]
df_youngs.to_excel('yates_statsmodels_young.xlsx', sheet_name = 'Sheet4')


######################################################################
#COATING SEPARATE OLS FITTING
#OLS regression on coating separate 51, 52, and 68, and normal 51, 52, and 68. 
######################################################################

lasso_df = pd.read_excel('lasso_data_table.xlsx')
del lasso_df['Unnamed: 0']
lasso_df.drop(lasso_df.iloc[:, 5:8], inplace=True, axis=1)
lasso_df.columns = ['sample_name', 'young_pa', 'elongation', 'toughness_pa','max_strength_pa', 'coat_sep', 'reinforcement', 'double_layer']

#toughness
mod_coat_sep_toughness_pa = smf.ols('toughness_pa~C(coat_sep)*C(reinforcement)*C(double_layer)', data=lasso_df)
res_coat_sep_toughness_pa = mod_coat_sep_toughness_pa.fit()

coat_sep_df_toughness_pa = pd.concat((res_coat_sep_toughness_pa.params, res_coat_sep_toughness_pa.bse, res_coat_sep_toughness_pa.tvalues, res_coat_sep_toughness_pa.pvalues), axis=1)
coat_sep_df_toughness_pa.columns = ['parameter (Pa)', 'standard error of effect (Pa)', 't_value', 'p_value' ]
coat_sep_df_toughness_pa.to_excel('yates_coat_sep_statsmodels_toughness_pa.xlsx', sheet_name = 'Sheet1')

#max strength 
mod_coat_sep_max_strength_pa = smf.ols('max_strength_pa~C(coat_sep)*C(reinforcement)*C(double_layer)', data=lasso_df)
res_coat_sep_max_strength_pa = mod_coat_sep_max_strength_pa.fit()

coat_sep_df_max_strength_pa = pd.concat((res_coat_sep_max_strength_pa.params, res_coat_sep_max_strength_pa.bse, res_coat_sep_max_strength_pa.tvalues, res_coat_sep_max_strength_pa.pvalues), axis=1)
coat_sep_df_max_strength_pa.columns = ['parameter (Pa)', 'standard error of effect (Pa)', 't_value', 'p_value' ]
coat_sep_df_max_strength_pa.to_excel('yates_coat_sep_statsmodels_max_strength_pa.xlsx', sheet_name = 'Sheet1')

#elongation at break 
mod_coat_sep_elongation = smf.ols('elongation~C(coat_sep)*C(reinforcement)*C(double_layer)', data=lasso_df)
res_coat_sep_elongation = mod_coat_sep_elongation.fit()

coat_sep_df_elongation = pd.concat((res_coat_sep_elongation.params, res_coat_sep_elongation.bse, res_coat_sep_elongation.tvalues, res_coat_sep_elongation.pvalues), axis=1)
coat_sep_df_elongation.columns = ['parameter (%)', 'standard error of effect (%)', 't_value', 'p_value' ]
coat_sep_df_elongation.to_excel('yates_coat_sep_statsmodels_elongation.xlsx', sheet_name = 'Sheet1')

#young's modulus 
mod_coat_sep_youngs_pa = smf.ols('young_pa~C(coat_sep)*C(reinforcement)*C(double_layer)', data=lasso_df)
res_coat_sep_youngs_pa = mod_coat_sep_youngs_pa.fit()

coat_sep_df_youngs_pa = pd.concat((res_coat_sep_youngs_pa.params, res_coat_sep_youngs_pa.bse, res_coat_sep_youngs_pa.tvalues, res_coat_sep_youngs_pa.pvalues), axis=1)
coat_sep_df_youngs_pa.columns = ['parameter (Pa)', 'standard error of effect (Pa)', 't_value', 'p_value' ]
coat_sep_df_youngs_pa.to_excel('yates_coat_sep_statsmodels_youngs_pa.xlsx', sheet_name = 'Sheet1')
