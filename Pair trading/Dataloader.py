#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 13:00:54 2025

@author: shayannikkar
"""

import pandas as pd
import yfinance as yf 
import datetime as dt
import itertools
from statsmodels.tsa.stattools import coint


def load_data():
    
    '''
    Arg : No input , downloading close price of stocks from yfinance
    
    Return : A dataframe consists of close price of the stocks chosen
    '''
    
    tickers = tech_stocks = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
        "NFLX", "ADBE", "CRM", "INTC", "AMD", "QCOM", "CSCO",
        "ORCL", "IBM", "PYPL", "UBER", "SNAP", "SHOP"
    ]
    
    start = "2022-01-01"
    end = dt.datetime.today()
    data = pd.DataFrame()
    
    for ticker in tickers:
        try:
            df = yf.download(ticker , start = start , end = end , interval="1d")
            data[ticker] = df["Close"]
            
        except FileNotFoundError():
            print("Ticker {} is not available in Yahoo Finance!".format(ticker))
        
    return data



def pair_search(data):
    
    
    '''
    Arg: Dataframe consist of stocks closing prices
    
    Return : DataFrame consist of good pairs for pair trading with Lowest p_value(best stocks)
    to Highest p_value(less correlated stocks)
    
    
    '''
    
    df = data.copy()
    
    pair_list = list(itertools.combinations(df.columns , 2))
    highly_correlated_pairs = []
    
    for stock1 , stock2 in pair_list:
        score , p_value , _ = coint(df[stock1] , df[stock2])
        print(f"Testing pair {stock1} & {stock2} : p_value is {p_value:.3f}")
        print("\n")
        
        if p_value <= 0.1:
            highly_correlated_pairs.append((stock1 , stock2 , p_value))
            print(f"{stock1} and {stock2} are good pair")
            print("*******************************************************************")
            
    if highly_correlated_pairs:
        result_df = pd.DataFrame(highly_correlated_pairs , columns= ["Stock 1" , "Stock 2" ,
                                                                     "P_value"])
        result_df.sort_values("P_value" , ascending=True , inplace=True)
    else:
        print("No Good pair Found!!!!")
        return
    
    return result_df
    


