#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 14:08:31 2018

@author: Ling
"""

"""
Created on Sat May 19 23:18:30 2018

@author: Ling
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import holtwinters as ht
import datetime as dt

#########
#M odeling by all sample data 
# Load the dataset; Data from Jan 1998 to Mar 2018
visitors = pd.read_csv('InternationalArrivals.csv')

# Seperate the time and variable y
Y_a=numVisitors = visitors[visitors.columns[1]]
X_a= months = visitors[visitors.columns[0]]

# Training set Jan 1998 - Dec 2015 & test set Jan 2016 - Mar 2018
training = numVisitors[:216]
test = numVisitors[216:]

months_tr = months[:216]
months_test = months[216:]

# Product time for plotting purpose
x_a = np.array([dt.datetime.strptime(d, '%b-%y') for d in months]) 
x_tr = np.array([dt.datetime.strptime(d, '%b-%y') for d in months_tr]) 
x_test = np.array([dt.datetime.strptime(d, '%b-%y') for d in months_test]) 

# Prediction Year
xp = np.array([dt.datetime.strptime(d, '%b-%y') for d in ('Apr-18','May-18','Jun-18','Jul-18','Aug-18','Sep-18','Oct-18','Nov-18','Dec-18','Jan-19','Feb-19','Mar-19')]) 
xp_a = np.hstack((x_a,xp))


# Plot the original test set 
plt.figure(figsize=(12,6))
plt.plot(x_a, numVisitors, label= 'observed')
plt.title('International International Arrivals from Jan 1998 to Mar 2018')
plt.xlabel('Time')
plt.ylabel('No of Visitors')
plt.legend(loc=2) 

# 4 sarima
import statsmodels as sm 
import statsmodels.api as smt
from pandas.tools.plotting import autocorrelation_plot
from datetime import datetime

#check stationary for the original data 
#original data
smt.graphics.tsa.plot_acf(Y_a.values,lags=30,alpha = 0.05)
plt.title('ACF:Original Data')
smt.graphics.tsa.plot_pacf(Y_a.values,lags=30,alpha = 0.05)
plt.title('PACF:Original Data')
# non-stationary 

#take log (as magnitude is gradually increasing, like multiplicative method)
Y_a_log = np.log(Y_a.values)

# best Sarima(1,1,4)(2,1,0)12, order from training test
sarima_model_a= smt.tsa.statespace.SARIMAX(Y_a_log, order=(1,1,4),seasonal_order=(2,1,0,12)) 
# Estimating the model
result_a= sarima_model_a.fit(disp=False)
print(result_a.summary())

forecasts_sarima_a = result_a.forecast(12)
predicted_4_a = np.exp(forecasts_sarima_a)

# Display forecasting on log data
fig = plt.figure(figsize=(12,6)) 
plt.plot(x_a,Y_a_log,label='log transformed data')
plt.plot(xp,forecasts_sarima_a,label='forecast by SARIMA on log data')
plt.title('International Arrivals Forecast for Apr 2018 - Mar 2019 by SARIMA(1,1,4)(2,1,0)12' )
plt.legend(loc=2)

# Display forecasting on original data
fig = plt.figure(figsize=(12,6)) 
plt.plot(x_a,Y_a,label='observed')
plt.plot(xp,predicted_4_a,label='forecast by SARIMA on original data')
plt.title('International Arrivals Forecast for Apr 2018 - Mar 2019 by SARIMA(1,1,4)(2,1,0)12' )
plt.legend(loc=2)

fig = plt.figure(figsize=(12,6)) 
plt.plot(xp,predicted_4_a,label='forecast by SARIMA on original data')
plt.title('International Arrivals Forecast for Apr 2018 - Mar 2019 by SARIMA(1,1,4)(2,1,0)12' )
plt.legend(loc=2)
