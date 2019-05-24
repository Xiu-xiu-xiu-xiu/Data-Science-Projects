
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# PART 1 EDA on the whole data set
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import holtwinters as ht
import datetime as dt

# EDA on original data 
# Load the dataset;  
visitors = pd.read_csv('InternationalArrivals.csv')

# Seperate the time and variable y
numVisitors = visitors[visitors.columns[1]]
months = visitors[visitors.columns[0]]

# Product time for plotting purpose
#%y stands for years（00-99; %b stands for months
x = np.array([dt.datetime.strptime(d, '%b-%y') for d in months]) 

# Plot the original data
plt.figure(figsize=(12,6))
plt.plot(x,numVisitors,label = 'observed')
plt.title('International Arrivals from Jan 1998 to Mar 2018')
plt.xlabel('Time')
plt.ylabel('No of Visitors')
plt.legend(loc=2) 

# Highlight the seasonal component peaks
plt.figure(figsize=(12,6))
plt.plot(x,np.power(numVisitors, 2) )
plt.title('Power 2 - International Arrivals from Jan 1998 to Mar 2018')
plt.xlabel('Time')
plt.ylabel('No of Visitors')

# To calculate the initial trend-cycle estimate we need to do moving average smoothing
# m=12, 2x12 MA by chaining a 2-MA and a 12-MA as follows:
T = numVisitors.rolling(2, center = True).mean().rolling(12,center = True).mean()
T = T.shift(-1)

# Plot the initial trend estimate
plt.figure(figsize=(12,6))
plt.plot(x,T)
plt.title('Initial TREND estimate - International Arrivals from Jan 1998 to Mar 2018s')
plt.xlabel('Time')
plt.ylabel('No of Visitors')

# Calculate the detrend series
S_multiplicative = numVisitors / T

# Plot the the detrend series
plt.figure(figsize=(12,6))
plt.plot(x, S_multiplicative)
plt.title('detrend series - International Arrivals from Jan 1998 to Mar 2018')
plt.xlabel('Time')
plt.ylabel('No of Visitors')

# Calculate seasonally adjusted data
safe_S = np.nan_to_num(S_multiplicative)
monthly_S = np.reshape(np.concatenate( (safe_S,[0,0,0,0,0,0,0,0,0]), axis = 0), (21,12))
monthly_avg = np.mean(monthly_S[1:19,:], axis=0)

# Constant C, M =12
c = 12 / np.sum(monthly_avg)

# Montnly indicies, which is the seasonal component St
S_bar = c * monthly_avg

# Repeat the average over 21 years
tiled_avg = np.tile(S_bar, 21)

# Plot seasonal component 
plt.figure(figsize=(12,6))
plt.plot(x,tiled_avg[:243])
plt.title('Seasonal Component - Multiplicative Model')
plt.xlabel('Time')

# Seasonally adjusted series 
seasonally_adjusted = numVisitors / tiled_avg [:243]

# Plot seasonally adjusted series
plt.figure(figsize=(12,6))
plt.plot(x,numVisitors, label = "observed")
plt.plot(x,seasonally_adjusted,label = 'seasonally adjusted')
plt.title('Seasonally adjusted series - International Arrivals from Jan 1998 to Mar 2018')
plt.xlabel('Time')
plt.ylabel('No of Visitors')
plt.legend()

# seasonally adjusted series regress with time
plt.figure(figsize=(12,6))
plt.title("seasonally adjusted series regress with time")
plt.scatter(x,seasonally_adjusted)
plt.xlabel("Month")
plt.ylabel("No of Visitors")
plt.show(block=False)

# use polynomail with degree = 3 to fit the trend 
import numpy as np

X_poly = np.arange(1, len(seasonally_adjusted)+1, 1)
Y_poly = seasonally_adjusted


polymodel_fit = np.polyfit(X_poly,Y_poly,3)
polymodel = np.poly1d(polymodel_fit)

# trend formula coefficient 
print(polymodel_fit )

# trend formula 
print(polymodel ) 

x_polypred = np.arange(1, len(seasonally_adjusted)+1, 1)

poly_pred= np.polyval(polymodel_fit,x_polypred)

#trend prediction
trend_poly = poly_pred  

# Plot the final trend by polynomail model (power = 3)
plt.figure(figsize=(12,6))
plt.plot(x,numVisitors, label = "observed")
plt.plot(x,seasonally_adjusted,label = 'seasonally adjusted')
plt.plot(x,trend_poly, label="Trend")
plt.legend()
plt.title("Final TREND estimate - International Arrivals from Jan 1998 to Mar 2018")
plt.xlabel("Time")
plt.ylabel("No of Visitors")
plt.show(block=False)

# Predicted by trend * seasonality
prediction_ploy_seasonal = poly_pred * tiled_avg [:243]

# Plot decomposition method (trend and seasonality) 
plt.figure(figsize=(12,6))
plt.plot(x,numVisitors,label='observed')
plt.plot(x,prediction_ploy_seasonal, label='multiplicative decomposition method')
plt.legend(loc=2)
plt.title("International Arrivals from Jan 1998 - Mar 2018 by Decomposition Method")
plt.xlabel("Month")
plt.ylabel("No of Visitor")

# scatter plot 
# The residuals are fluctuated around 1
# The multiplicative model fits well 
plt.figure(figsize=(12,6))
plt.title("Residual Plot")
plt.plot(x, seasonally_adjusted / trend_poly)
plt.xlabel("Residuals")
plt.ylabel('Time')
plt.show(block=False)










