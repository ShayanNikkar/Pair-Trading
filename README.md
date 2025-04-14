
# Pair-Trading
in this project we search between given tickers to find out which stocks are highly correlated , defining a pair trading strategy to produce spread and learn and predict the spread by xgboostregressor to produce signals.
## 📚Project Overview
#### Data Handling:
- *Define a list of tickers.*
- *Download historical price data from Yahoo Finance.*
#### Cointegration:
- *Test pairs of tickers using cointegration analysis.*
- *Identify highly correlated pairs suitable for pair trading.*
#### Spread Calculation & Prediction:
- *Define the spread between the two stocks in a pair.*
- *Use XGBoost regressor to learn and predict the future spread.*
- *Generate trading signals based on the mean-reversion assumption of the spread.*
#### Visualization:
- *Visualize price charts for individual stocks.*
- *Display spread charts, including both the test and predicted spread.*
- *Plot the cumulative return of the strategy for each pair.*
#### Performance Evaluation:
- *Calculate performance metrics for each pair on the test period:*
- *Sharpe Ratio*
- *Average Return*
- *Geometric Return*
- *Standard Deviation*

## 📦package requirements
pandas  
numpy  
statsmodels  
sklearn  
yfinance  
datetime  
matplotlib  
itertools
