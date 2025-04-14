#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 14:50:43 2025

@author: shayannikkar
"""
import pandas as pd
from xgboost import XGBRegressor
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split,GridSearchCV



def OLS_spread(data):
    df = data.copy()
    if df.iloc[0 , 0] >= df.iloc[0 ,1]:
        H = df.iloc[:, 0]
        L = df.iloc[: , 1]
    else:
        H = df.iloc[: , 1]
        L = df.iloc[: , 0]
    X = sm.add_constant(L)
    ols = sm.OLS(H, X).fit()
    sp = H - ols.predict(X)
    sp.name = "spread"
    return sp , H , L



def model(data , window_size = 30):
    
    df = data.copy()
    sp , H , L = OLS_spread(df)
    df = pd.concat([df , sp] ,axis=1)
    
    '''
    Feature Eng
    '''
    df[f"{H.name}_ret"]= df[f"{H.name}"].pct_change()
    df[f"{L.name}_ret"]= df[f"{L.name}"].pct_change()
    df["spread_mean"] = df["spread"].rolling(window = window_size).mean()
    df["spread_std"] = df["spread"].rolling(window = window_size).std()
    df["z-score"] = (df["spread"]- df["spread_mean"]) / df["spread_std"]
    df["ret_interaction"] = df[f"{H.name}_ret"] * df[f"{L.name}_ret"]
    df["spread_lag1"] = df["spread"].shift(1)
    df["spread_lag2"] = df["spread"].shift(2)


    final_df = df.drop([f"{H.name}",f"{L.name}" ,
                        f"{H.name}_ret" , f"{L.name}_ret"] , axis=1)
    
    final_df["target"] = final_df["spread"].shift(-1)-final_df["spread"]
    final_df.dropna(inplace=True)
    print(f"Feature Engineering for {H.name} & {L.name} Done!")
    print("\n")


    X=final_df.drop("target" , axis=1)
    y = final_df["target"]
    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , shuffle=False)
    print("Start Fitting the model")
    xgb =XGBRegressor(tree_method = "exact" , random_state=42 , n_jobs=1)
    param_grid = {"max_depth" : [5,10,15],
                  "learning_rate" : [1]}
    xgb_reg = GridSearchCV(xgb, param_grid = param_grid ) 
    xgb_reg.fit(X_train , y_train)
    y_pred = xgb_reg.predict(X_test)
    print("Model Fitted Successfully!")
    
    
    df_trade = pd.DataFrame(y_pred , index = y_test.index , columns=["predicted_spread"] )
    df_ret = df[[f"{H.name}_ret" , f"{L.name}_ret"]].loc[y_test.index]
    df_trade = pd.concat([df_ret , df_trade] , axis=1)

    df_trade["signal"]=0
    pred_abs = np.abs(df_trade["predicted_spread"])
    threshold = np.percentile(pred_abs, 25)
    df_trade.loc[df_trade["predicted_spread"] > threshold , "signal"] = -1
    df_trade.loc[df_trade["predicted_spread"] < -threshold , "signal"] = 1
    df_trade["position"] = df_trade["signal"].shift(1)
    df_trade["strategy_ret"] = (df_trade[f"{H.name}_ret"] - df_trade[f"{L.name}_ret"]) * df_trade["position"]
    df_trade["actual_spread"] = final_df["target"].loc[y_test.index]
    df_trade["cum_ret"] = round(np.cumprod(1 + df_trade["strategy_ret"]),4)
    
    strategy_mean_ret = round(df_trade["strategy_ret"].mean(),4)
    strategy_std = round(df_trade["strategy_ret"].std(),4)
    strategy_geometric_ret = round(df_trade["cum_ret"][-1] - 1,4)
    strategy_SR_avr = round(strategy_mean_ret / strategy_std,4)
    strategy_SR_geo = round(strategy_geometric_ret/ strategy_std,4)
    metrics = pd.DataFrame({"Mean_Return":[strategy_mean_ret],
                           "Standard Deviation" : [strategy_std],
                           "Geometric_Return" : [strategy_geometric_ret],
                           "Average Sharp Ratio": [strategy_SR_avr],
                           "Geometric Sharp Ratio":[strategy_SR_geo]})
    


    print(f"Strategy Average return for trading {H.name} & {L.name} is :{strategy_mean_ret * 100}%")
    print(f"Strategy Standard Deviation is :{strategy_std *100}%")
    print(f"Strategy Geometric return is :{strategy_geometric_ret * 100}%")
    print(f"Strategy Average Sharp Ratio  is :{strategy_SR_avr}")
    print(f"Strategy Geometric Sharp Ratio  is :{strategy_SR_geo}")
    
    
    return df_trade , metrics
    
         


