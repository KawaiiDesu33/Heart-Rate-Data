#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 15:53:19 2022

@author: terenceau
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf

path = '/Users/terenceau/Desktop/Python/Heart Data/heart.csv'
data = pd.read_csv(path)
headers = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholestrol',
           'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina'
           , 'Oldpeak', 'ST_Slope', 'HeartDisease']

data.columns = headers

### General Relationship Plots
# -- Age

age_BP = sns.relplot(kind = 'scatter', x = 'Age', y = 'RestingBP', hue = 'Sex'
                     , data = data)
age_HR = sns.relplot(kind = 'scatter', x = 'Age', y = 'MaxHR', hue = 'Sex'
                     , data = data)
age_Cho = sns.relplot(kind = 'scatter', x = 'Age', y = 'Cholestrol', hue = 'Sex'
                     , data = data)

# -- Cholestrol
Cho_BP = sns.relplot(kind = 'scatter', x = 'Cholestrol', y = 'RestingBP',
                     hue = 'Sex', data = data)
Cho_HR = sns.relplot(kind = 'scatter', x = 'MaxHR', y = 'RestingBP',
                     hue = 'Sex', data = data)

# -- Heart Rate
HR_BP = sns.relplot(kind = 'scatter', x = 'RestingBP', y = 'MaxHR'
                    , hue = 'Sex', data = data)

### Dummy Variables for Qualitative Data
sex_dummy = pd.get_dummies(data['Sex'])
temp = ['Females', 'Males']
sex_dummy.columns = temp

exer_dummy = pd.get_dummies(data['ExerciseAngina'])
temp = ['Exercise_False', 'Exercise_True']
exer_dummy.columns = temp

data['ChestPainType'].describe()
data['RestingECG'].describe()
data['ExerciseAngina'].describe()
data['ST_Slope'].describe()

data_tot = data.join(sex_dummy)
data_tot = data_tot.join(exer_dummy)

### Listing Unique Values
print(data['Sex'].unique())
print(data['ChestPainType'].unique())
print(data['RestingECG'].unique())
print(data['ExerciseAngina'].unique())
print(data['ST_Slope'].unique())

### General Correlation of Data
data_corr = data.corr()

### Linear Regressions
#LPM - OLS Model
lpm_model = smf.ols(formula = 'HeartDisease ~ Males + Exercise_True + RestingBP + Cholestrol + FastingBS + MaxHR'
                    , data = data_tot)
lpm_results = lpm_model.fit()
print(lpm_results.summary())

lpm_model2 = smf.ols(formula = 'HeartDisease ~ C(Sex) + C(ExerciseAngina) + RestingBP + Cholestrol + FastingBS + MaxHR'
                    , data = data_tot)
lpm_results2 = lpm_model2.fit()
print(lpm_results2.summary())

# - Same Results for Using onle C ( ) Method
lpm_model3 = smf.ols(formula = 'HeartDisease ~ C(Sex) + C(ChestPainType) + RestingBP + Cholestrol + FastingBS + C(RestingECG) + MaxHR + C(ExerciseAngina) + C(ST_Slope)'
                     , data = data)
lpm_result3 = lpm_model3.fit()
print(lpm_result3.summary())

### Logistic Model
logit_model = smf.logit(formula = 'HeartDisease ~ C(Sex) + C(ChestPainType) + RestingBP + Cholestrol + FastingBS + C(RestingECG) + MaxHR + C(ExerciseAngina) + C(ST_Slope)'
                     , data = data)
logit_results = logit_model.fit()
print(logit_results.summary())

# - Marginal Effects
logit_marg = logit_results.get_margeff()
print(logit_marg.summary())


### Probit Model
probit_model = smf.probit(formula = 'HeartDisease ~ C(Sex) + C(ChestPainType) + RestingBP + Cholestrol + FastingBS + C(RestingECG) + MaxHR + C(ExerciseAngina) + C(ST_Slope)'
                     , data = data)
probit_results = probit_model.fit()
print(probit_results.summary())

# - Marginal Effects
probit_marg = probit_results.get_margeff()
print(probit_marg.summary())


### Diagnostic Graphs - lpm_result3
figure = sm.graphics.plot_partregress_grid(lpm_result3)

figure2 = sm.graphics.plot_fit(lpm_result3, 'RestingBP')

figure3 = sm.graphics.plot_regress_exog(lpm_result3, 'RestingBP')
figure3.tight_layout(pad=1.0)

figure4 = sm.graphics.plot_leverage_resid2(lpm_result3, 'RestingBP')





















