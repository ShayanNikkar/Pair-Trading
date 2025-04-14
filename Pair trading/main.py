#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 17:46:10 2025

@author: shayannikkar
"""
import os
os.chdir("/Users/shayannikkar/Desktop/Qmul/Projects/Pair trading")

import numpy as np
np.random.seed(42)

from Model import model , OLS_spread
from Dataloader import load_data , pair_search
from Plotter import Plotter
import pandas as pd

def main():
    data = load_data()
    good_pairs = pair_search(data)    
    Result = pd.DataFrame()

    for i in range(len(good_pairs)):
        stock1,stock2 = good_pairs.iloc[i , 0:2]
        df = data.loc[: , [stock1,stock2]]
        df_trade , metrics= model(df)
        metrics.index = [f"{stock1}–{stock2}"]
    
        y_test = df_trade["actual_spread"]
        y_pred=df_trade["predicted_spread"] 
        cum_return = df_trade["cum_ret"]
    
        plot_pair = Plotter(df)
        plot_pair.plot_price()
        plot_pair.plot_real_spread()
        plot_pair.plot_spread()
        plot_pair.plot_actual_vs_prediction(y_test, y_pred)
        plot_pair.plot_cum_ret(cum_return)
    
        Result = pd.concat([Result , metrics] , axis=0)
        
    return Result

if __name__ == "__main__":
    result = main()

