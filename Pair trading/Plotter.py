#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 13:54:50 2025

@author: shayannikkar
"""

import matplotlib.pyplot as plt
from Model import OLS_spread



class Plotter():
    def __init__(self , data):
        self.data = data
        self.stock1 = data.iloc[: , 0]
        self.stock2 = data.iloc[: , 1]
        self.ticker1 = self.stock1.name
        self.ticker2 = self.stock2.name
        
    def plot_price(self):
        plt.plot(figsize=(10,8))
        plt.plot( self.stock1 , label = self.ticker1 , color='r')
        plt.plot(self.stock2 , label = self.ticker2 , color='navy')
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title("Price chart of {} & {}".format(self.ticker1 , self.ticker2))
        plt.legend()
        plt.grid()
        plt.show()
        
    def plot_real_spread(self):
        if self.stock1[0]> self.stock2[0]:
            spread = self.stock1 - self.stock2
        else:
            spread = self.stock2 - self.stock1
            
        plt.plot(figsize=(10,8))
        plt.plot( spread , label = f"{self.ticker1} - {self.ticker2}" , color='blue')
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title("Spread chart of {}–{}".format(self.ticker1 , self.ticker2))
        plt.legend()
        plt.grid()
        plt.show()
        
    def plot_spread(self):
        sp , _ , _ = OLS_spread(self.data)
        plt.plot(figsize=(10,8))
        plt.plot( sp , label = f"{self.ticker1} - {self.ticker2}" , color='green')
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title(" OLS Spread chart of {}–{}".format(self.ticker1 , self.ticker2))
        plt.legend()
        plt.grid()
        plt.show()
        
        
    def plot_actual_vs_prediction(self , y_test , y_pred):
        plt.plot(figsize=(10,8))
        plt.plot( y_test.index , y_test , label = "Actual" , color='blue')
        plt.plot( y_test.index , y_pred , label = "prediction" , color='red')
        plt.xlabel("Date")
        plt.ylabel("spread.shift(-1) - spread")
        plt.title("Actual vs Prediction chart of {}–{}".format(self.ticker1 , self.ticker2))
        plt.legend()
        plt.grid()
        plt.show()
        
    def plot_cum_ret(self , cumret):
        plt.plot(figsize=(10,8))
        plt.plot( cumret , label = f"{self.ticker1} - {self.ticker2}" , color='black')
        plt.xlabel("Date")
        plt.ylabel("Cum Return")
        plt.title("Cumulative Return of {}–{} Strategy".format(self.ticker1 , self.ticker2))
        plt.legend()
        plt.grid()
        plt.show()
        
        
        
    def __str__(self):
        return self.ticker1 + '_' + self.ticker2 + "Plotting!"
        
        
        





