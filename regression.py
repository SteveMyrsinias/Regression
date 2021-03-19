# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 10:00:56 2021

@author: smirs
"""
#TESTESTESTEST
#Import Libraries
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
#from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import  stats 
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

# import train data
df_train = pd.read_csv('train.csv')

df_train.head()
df_train.columns
df_train['SalePrice'].describe()

#histogram
sns.distplot(df_train['SalePrice']);


#relationship of square space and sales price
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
data.plot.scatter(x=var, y='SalePrice',ylim=(0.000000))
print(data[:10])

# saleprice correlation matrix
corrmat = df_train.corr()
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index   # nlargest : pick the most powerfull correlation
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent], axis=1, keys=['Total','Percent'])
missing_data.head(20)

#dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max()

# standardizing data
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]  # rearrange ascending
high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]  # rearrange descending
print('outer range (low) of the distribution : ')
print(low_range)
print('\nouter range (high) of the distribution : ')
print(high_range)

#bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

# dealing point 
df_train.sort_values(by = 'GrLivArea', ascending=False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
#bivariate analysis saleprice/grlivarea
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

# applying log transformation
df_train['SalePrice'] = np.log(df_train['SalePrice'])
# histogram and normal probability plot 
sns.distplot(df_train['SalePrice'],fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)

# applying log transformation
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
# histogram and normal probability plot 
sns.distplot(df_train['GrLivArea'],fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)

# create column for new variale (one is enough because it's a binary categorical feature )
# if area > 0 it gets 1, for area ==0 it gets 0
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index = df_train.index)
df_train['HasBsmt'] = 0
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1

print(df_train['TotalBsmtSF'][:10])
print(df_train['HasBsmt'][:10])

# transform data
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
# histogram and normal probabiloty plot
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)

#scatter plot
plt.scatter(df_train['GrLivArea'], df_train['SalePrice']);

# scatter plot 
plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice']);

# convert categorical variable into dummy
df_train = pd.get_dummies(df_train)
df_train.head()
df_train['SalePrice']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X = df_train.drop('SalePrice', axis=1)
y = df_train['SalePrice']
X.head()
y.head()

#Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20, shuffle =True)

from sklearn.linear_model import LinearRegression
rgr = LinearRegression()
rgr.fit(X_train, y_train)
# Evaluating the model
print('Reggression Train Score is : ' , rgr.score(X_train, y_train)*100)
print('Reggression Test Score is : ' , rgr.score(X_test, y_test)*100)
# Predicting the Test set results
y_pred = rgr.predict(X_test)
mean_absolute_error(y_test, y_pred)
print('Mena Squard Error IS :     ',mean_squared_error(y_test, y_pred))
print('Mean Absolute Error Is :   ',mean_absolute_error(y_test, y_pred))
print('Median Absolute Error Is : ',median_absolute_error(y_test, y_pred))
# Function to invert target variable array from log scale
def inv_y(transformed_y):
    return np.exp(transformed_y)

# Series to collect RMSE for the different algorithms: "algortihm name + RMSE"
rmse_compare = pd.Series()
rmse_compare.index.name = "Model"

# Series to collect the accuracy for the different algorithms: "algorithms name + score"
scores_compare = pd.Series()
scores_compare.index.name = "Model"

# Model 1: Linear Regression =======================
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_val_predictions = linear_model.predict(X_test)
linear_val_rmse = mean_squared_error(inv_y(linear_val_predictions), inv_y(y_test))
linear_val_rmse = np.sqrt(linear_val_rmse)
rmse_compare['LinearRegression'] = linear_val_rmse

lr_score = linear_model.score(X_test, y_test)*100
scores_compare['LinearRegression'] = lr_score

#Model 2: Decision Tress ===========================
dtree_model = DecisionTreeRegressor(random_state=5)
dtree_model.fit(X_train, y_train)

dtree_val_predictions = dtree_model.predict(X_test)
dtree_val_rmse = mean_squared_error(inv_y(dtree_val_predictions), inv_y(y_test))
dtree_val_rmse = np.sqrt(dtree_val_rmse)
rmse_compare['DecisionTreeRegressor'] = dtree_val_rmse

dtree_score = dtree_model.score(X_test, y_test)*100
scores_compare['DecisionTreeRegressor'] = dtree_score

# Model 3: Random Forest ==========================
rf_model = RandomForestRegressor(random_state=5)
rf_model.fit(X_train, y_train)

rf_val_predictions = rf_model.predict(X_test)
rf_val_rmse = mean_squared_error(inv_y(rf_val_predictions), inv_y(y_test))
rf_val_rmse = np.sqrt(rf_val_rmse)
rmse_compare['RandomForest'] = rf_val_rmse

rf_score = rf_model.score(X_test, y_test)*100
scores_compare['RandomForest'] = rf_score


# Model 4: Gradient Boostinf Regression ===========
gbr_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=5)
gbr_model.fit(X_train, y_train)

gbr_val_predictions = gbr_model.predict(X_test)
gbr_val_rmse = mean_squared_error(inv_y(gbr_val_predictions), inv_y(y_test))
gbr_val_rmse = np.sqrt(gbr_val_rmse)
rmse_compare['GradientBoostingRegression'] = gbr_val_rmse

gbr_score = gbr_model.score(X_test, y_test)*100
scores_compare['GradientBoostingRegression'] = gbr_score

print("RMSE values for different algorithms:")
rmse_compare.sort_values(ascending=True).round()
print("Accuracy scores for different algorithms")
scores_compare.sort_values(ascending=False).round(3)