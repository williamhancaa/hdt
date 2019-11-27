# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:38:57 2019

@author: william.han
"""

import pandas as pd
import numpy as np
import sys 
sys.path.append(r'C:\Users\william.han\Desktop\HDT')
import visuals as vs
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
from time import time
from sklearn.metrics import f1_score, accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso,LassoCV
from sklearn.linear_model import Ridge,RidgeCV
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

def distribution(data, col_name,transformed = False):
    """
    Visualization code for displaying skewed distributions of features
    """
    
    # Create figure
    fig = pl.figure(figsize = (11,5));

    # Skewed feature plotting
    for i, feature in enumerate([col_name]):
        ax = fig.add_subplot(1, 2, i+1)
        ax.hist(data[feature], bins = 25, color = '#00A0A0')
        ax.set_title("'%s' Feature Distribution"%(feature), fontsize = 14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        ax.set_ylim((0, 2000))
        ax.set_yticks([0, 500, 1000, 1500, 2000])
        ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])

    # Plot aesthetics
    if transformed:
        fig.suptitle("Log-transformed Distributions of Continuous Census Data Features", \
            fontsize = 16, y = 1.03)
    else:
        fig.suptitle("Skewed Distributions of Continuous Census Data Features", \
            fontsize = 16, y = 1.03)

    fig.tight_layout()
    fig.show()




file_path = r'C:\Users\william.han\Desktop\HDT\Off_leasing_test.xlsx'
df = pd.read_excel(file_path, sheet_name ='result')
df['Obj_Dailyrent'] = df['Obj_Dailyrent'].apply(lambda x: np.log(x + 1))
df['Leasable_Area'] = df['Leasable_Area'].apply(lambda x: np.log(x+1))
df['Dis_subway'] = df['Dis_subway'].apply(lambda x: np.log(x+1))
df['Dis_bus'] = df['Dis_bus'].apply(lambda x: np.log(x+1))
df['Dis_School'] = df['Dis_School'].apply(lambda x: np.log(x+1))
df['Dis_Hospital'] = df['Dis_Hospital'].apply(lambda x: np.log(x+1))
df['Ttlfloor'] = df['Ttlfloor'].apply(lambda x: np.log(x+1))
col_name_list = [item for item in df.columns]

# Initialize a scaler, then apply it to the features
scaler = preprocessing.MinMaxScaler() # default=(0, 1)
numerical = ['Leasable_Area', 'Dis_subway',  'Dis_bus', 'Dis_School','Dis_Hospital','Mngfee','Ttlfloor']
df_transform = df
df_transform['No.of_School'].replace('status problem',0,inplace=True)
df_transform[numerical] = scaler.fit_transform(df_transform[numerical])
df_transform = df_transform.drop(['Propname','Floor','Zipcode','YearBuilt','Property_developer','ClearHeight','TotalFloors','FloorAreaRatio',
                                  'Fac_ElevatorCounts','FloorArea','GrossFloorArea',], axis =1)
rental_df = df_transform['Obj_Dailyrent']
features_df = df_transform.drop(['Obj_Dailyrent'], axis =1)

features_df[['No.of_subway' ,'No.of_bus','No.of_ATM','No.of_School','No.of_Hospital','No.of_Camparable','No.of_restaurant'
             ,'No.of_stores','No.of_dailyls','No.of_sport','No.of_hotels','No.of_enterprises']] = features_df[['No.of_subway' ,'No.of_bus','No.of_ATM','No.of_School','No.of_Hospital','No.of_Camparable','No.of_restaurant','No.of_stores','No.of_dailyls',
                                                                                              'No.of_sport','No.of_hotels','No.of_enterprises']].astype(str)
#one-hot cate-features 
features_final =  pd.get_dummies(features_df)


# Split the 'features' and 'rent' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    rental_df, 
                                                    test_size = 0.2, 
                                                  random_state = 0)

# pca 
from sklearn.decomposition import PCA
pca = PCA(n_components=262)
principalComponents = pca.fit_transform(features_final)

# Split the 'features' and 'rent' data into training and testing after pca
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(principalComponents, 
                                                    rental_df, 
                                                    test_size = 0.2, 
                                                  random_state = 0)
#linear regreesion 
linreg = LinearRegression()
linreg.fit(X_train_pca,y_train_pca)
print(linreg.score(X_test_pca,y_test_pca))


#lasson 模型
model = LassoCV()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
print('最佳的alpha：',model.alpha_)  

 
#Ridge模型
model = RidgeCV()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
print('最佳的alpha：',model.alpha_)  

#xgbost
model_test = xgb.XGBRegressor()
model_test.fit(X_train,y_train.values.ravel(),verbose=True)
preds_train = model_test.predict(X_train)
metrics.r2_score(y_train,preds_train)
model_test.get_params()
#optim xgboost
param_test1 = {
        'max_depth':range(3,10,2),
        'min_child_weight':range(1,6,2)
        }
gsearch1 = GridSearchCV(estimator =xgb.XGBRegressor(base_score= 0.5,
 booster= 'gbtree',
 colsample_bylevel=1,
 colsample_bynode=1,
 colsample_bytree= 1,
 gamma= 0,
 importance_type= 'gain',
 learning_rate= 0.1,
 max_delta_step= 0,
 missing= None,
 n_estimators= 100,
 n_jobs= 1,
 nthread= None,
 objective= 'reg:linear',
 random_state= 0,
 reg_alpha= 0,
 reg_lambda= 1,
 scale_pos_weight= 1,
 seed= None,
 silent= None,
 subsample= 1,
 verbosity= 1), param_grid =param_test1, scoring='r2', cv=5 )
gsearch1.fit(X_train,y_train)
best_param1 = gsearch1.best_params_
best_score1 = gsearch1.best_score_

#try again
param_test2 = {
        'max_depth':[8,9,10],
        'min_child_weight':[1,2,3]
        }
gsearch2 = GridSearchCV(estimator =xgb.XGBRegressor(base_score= 0.5,
 booster= 'gbtree',
 colsample_bylevel=1,
 colsample_bynode=1,
 colsample_bytree= 1,
 gamma= 0,
 importance_type= 'gain',
 learning_rate= 0.1,
 max_delta_step= 0,
 missing= None,
 n_estimators= 100,
 n_jobs= 1,
 nthread= None,
 objective= 'reg:linear',
 random_state= 0,
 reg_alpha= 0,
 reg_lambda= 1,
 scale_pos_weight= 1,
 seed= None,
 silent= None,
 subsample= 1,
 verbosity= 1), param_grid =param_test2, scoring='r2', cv=5 )
gsearch2.fit(X_train,y_train)
best_param2 =gsearch2.best_params_
best_score2 =gsearch2.best_score_

#try 3 times
param_test3 = {
        'gamma':range(0,20,1),
        'colsample_bylevel':np.range(0.001,1,0.001),
        'colsample_bynode':np.range(0.001,1,0.001),
        'colsample_bytree': np.range(0.001,1,0.001)
        }
gsearch3 = GridSearchCV(estimator =xgb.XGBRegressor(base_score= 0.5,
 booster= 'gbtree',
 importance_type= 'gain',
 learning_rate= 0.1,
 max_delta_step= 0,
 missing= None,
 n_estimators= 100,
 n_jobs= 1,
 nthread= None,
 objective= 'reg:linear',
 random_state= 0,
 reg_alpha= 0,
 reg_lambda= 1,
 scale_pos_weight= 1,
 seed= None,
 silent= None,
 subsample= 1,
 verbosity= 1,
 max_depth = 9,
 min_child_weight =1
), param_grid =param_test3, scoring='r2', cv=5 )
gsearch3.fit(X_train,y_train)
best_param3 =gsearch3.best_params_
best_score3 =gsearch3.best_score_



#test optimzed parameters
model = xgb.XGBRegressor(base_score= 0.5,
 booster= 'gbtree',
 colsample_bylevel=1,
 colsample_bynode=1,
 colsample_bytree= 1,
 gamma= 0,
 importance_type= 'gain',
 learning_rate= 0.1,
 max_delta_step= 0,
 missing= None,
 n_estimators= 100,
 n_jobs= 1,
 nthread= None,
 objective= 'reg:linear',
 random_state= 0,
 reg_alpha= 0,
 reg_lambda= 1,
 scale_pos_weight= 1,
 seed= None,
 silent= None,
 subsample= 1,
 verbosity= 1,
 max_depth=9,
 min_child_weight=1
        )
model.fit(X_train,y_train.values.ravel(),verbose=True)
preds_train = model.predict(X_train)
metrics.r2_score(y_train,preds_train)
