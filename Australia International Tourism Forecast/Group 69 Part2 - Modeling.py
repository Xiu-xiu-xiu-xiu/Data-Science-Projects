#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
#PART 2 - Modeling based on test set

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import holtwinters as ht
import datetime as dt


#########
# Modeling 
# Load the dataset; Data from Jan 1998 to Mar 2018
visitors = pd.read_csv('InternationalArrivals.csv')

# Seperate the time and variable y
numVisitors = visitors[visitors.columns[1]]
months = visitors[visitors.columns[0]]

# Training set Jan 1998 - Dec 2015 & test set Jan 2016 - Mar 2018
training = numVisitors[:216]
test = numVisitors[216:]

months_tr = months[:216]
months_test = months[216:]

# Product time for plotting purpose
x_tr = np.array([dt.datetime.strptime(d, '%b-%y') for d in months_tr]) 
x_test = np.array([dt.datetime.strptime(d, '%b-%y') for d in months_test]) 

# Plot the original test set 
plt.figure(figsize=(12,6))
plt.plot(x_tr,training, label= 'observed')
plt.title('International International Arrivals from Jan 1998 to Mar 2018')
plt.xlabel('Time')
plt.ylabel('No of Visitors')
plt.legend(loc=2) 


#1 Drift Method (Base Model)
first_observation = training[0]
last_observation = training[215]
T = len(training)

# Prediction
predicted_1 = []
for i in range (0,27):
    h = i + 1
    predicted_1.append(last_observation + h * (last_observation - first_observation)/(T-1))

# Plot prediction 
fig1 = plt.figure(figsize=(12,6))
plt.plot(x_tr,training,label='observed')
plt.plot(x_test,predicted_1,'-r', label='forecast')
plt.title('Drift Method: International Arrivals forecast for Jan 2016 - Mar 2018')
plt.xlabel('Time')
plt.ylabel('No of Visitors')
plt.legend(loc=2) 


#2 Seasonal holtwinter multiplicative 
from holtwinters import multiplicative 
# variable, we need convert a Series to a python list
ts = training.tolist()

# Now we define 99 predictions we wish to predict
fc = 27

m = 12
x_smoothed, m, Y, alpha, beta, gamma, rmse = ht. multiplicative(ts,m,fc)

hh =x_smoothed[:-1]

# Prediction
predicted_2 = hh[216:]

# Plot prediction 
fig1 = plt.figure(figsize=(12,6))
plt.plot(x_tr,training,label='observed')
plt.plot(x_test,predicted_2,'-r', label='forecast')
plt.title('Seasonal Holts Multiplicative: International Arrivals forecast for Jan 2016 - Mar 2018')
plt.xlabel('Time')
plt.ylabel('No of Visitors')
plt.legend(loc=2) 


# 3 Decomposition Method - Multiplicative Method
# To calculate the initial trend-cycle estimate we need to do moving average smoothing
# m=12, 2x12 MA by chaining a 2-MA and a 12-MA as follows:
T_tr = training.rolling(2, center = True).mean().rolling(12,center = True).mean()
T_tr = T_tr.shift(-1)

# Plot the initial trend estimate
plt.figure(figsize=(12,6))
plt.plot(x_tr,T_tr, label = 'observed')
plt.title('Initial TREND estimate_training set - International Arrivals form Jan 1998 - Dec 2015')
plt.xlabel('Time')
plt.ylabel('No of Visitors')
plt.legend(loc=2) 

#Calculate seasonal components
S_multiplicative_tr = training / T_tr

#Then plot the detrend series
plt.figure(figsize=(12,6))
plt.plot(x_tr, S_multiplicative_tr)
plt.title('detrend series_training set - International Arrivals form Jan 1998 - Dec 2015')
plt.xlabel('Time')

#Calculate seasonally adjusted data
safe_S_tr= np.nan_to_num(S_multiplicative_tr)
monthly_S_tr= np.reshape(safe_S_tr,(18,12))
monthly_avg_tr= np.mean(monthly_S_tr[1:18,:], axis=0)

#constant C, M =12
c_tr= 12 / np.sum(monthly_avg_tr)

#montnly indicies, which is the seasonal component St
S_bar_tr = c_tr* monthly_avg_tr

#Repeat the average over 18 years
tiled_avg_tr= np.tile(S_bar_tr, 18)

#plot seasonal component 
plt.figure(figsize=(12,6))
plt.plot(x_tr,tiled_avg_tr[:216])
plt.title('Seasonal Component_training set - Short Term International Arrivals')
plt.xlabel('Time')

#seasonally adjusted series
seasonally_adjusted_tr = training / tiled_avg_tr 

#plot seasonally adjusted series
plt.figure(figsize=(12,6))
plt.plot(x_tr,training, label = "observed")
plt.plot(x_tr,seasonally_adjusted_tr,label = 'seasonally adjusted')
plt.title('Seasonally adjusted series_training set - International Arrivals from Jan 1998 to Dec 2015')
plt.xlabel('Time')
plt.ylabel('No of Visitors')
plt.legend(loc=2) 

# seasonally adjusted series reg with time
plt.figure(figsize=(12,6))
plt.title("seasonally adjusted series reg with time")
plt.scatter(x_tr,seasonally_adjusted_tr)
plt.xlabel("Month")
plt.ylabel("No of Visitors")
plt.show(block=False)

# use polynomail with degree = 3 to fit the trend 
import numpy as np

X_poly_tr= np.arange(1, len(seasonally_adjusted_tr)+1, 1)
Y_poly_tr= seasonally_adjusted_tr


polymodel_fit_tr = np.polyfit(X_poly_tr,Y_poly_tr,3)
polymodel_tr = np.poly1d(polymodel_fit_tr)

# trend formula coefficient matrix
print(polymodel_fit_tr ) 

# trend formula 
print(polymodel_tr )

x_polypred_tr = np.arange(1, len(seasonally_adjusted_tr)+1, 1)

poly_pred_tr= np.polyval(polymodel_fit_tr,x_polypred_tr)

# prediction made by trend on training set
trend_poly_tr = poly_pred_tr 

# Plot the final trend by polynomail model (power = 3)
plt.figure(figsize=(12,6))
plt.plot(x_tr,training, label = "observed")
plt.plot(x_tr,seasonally_adjusted_tr,label = 'seasonally adjusted')
plt.plot(x_tr,trend_poly_tr, label="Trend")
plt.legend()
plt.title("Final TREND estimate - International Arrivals from Jan 1998 to Dec 2015")
plt.xlabel("Time")
plt.ylabel("No of Visitors")
plt.show(block=False)

# Predicted by trend * seasonality
x_polypred_test = np.arange(len(seasonally_adjusted_tr), len(training)+1, 1)
poly_pred_test = np.polyval(polymodel_fit_tr,x_polypred_test)

prediction_ploy_seasonal_tr = poly_pred_test * tiled_avg_tr [:27]

predicted_3 = prediction_ploy_seasonal_tr.tolist()

# Plot the prediction by decomposition method 
plt.figure(figsize=(12,6))
plt.plot(x_tr,training,label='observed')
plt.plot(x_test,predicted_3, label='forecast by trend and seasonality')
plt.legend(loc=2)
plt.title("International Arrivals Forecast for Jan 2016 - Mar 2018 by Decomposition Method")
plt.xlabel("Month")
plt.ylabel("No of Visitor")

# The residuals is fluctuatd aroud 1
# The multiplicative model fits well 
plt.figure(figsize=(12,6))
plt.title("Residual Plot")
plt.plot(x_tr, seasonally_adjusted_tr / trend_poly_tr)
plt.xlabel("Residuals")
plt.ylabel('Time')
plt.show(block=False)


# 4 sarima
import statsmodels as sm 
import statsmodels.api as smt
from pandas.tools.plotting import autocorrelation_plot
from datetime import datetime

#check stationary for the original data 
#original data
smt.graphics.tsa.plot_acf(training.values,lags=30,alpha = 0.05)
plt.title('ACF:Original Data')
smt.graphics.tsa.plot_pacf(training.values,lags=30,alpha = 0.05)
plt.title('PACF:Original Data')
#ACF dies down slowly, non-stationary 

#take log (as magnitude is gradually increasing, like multiplicative method)
training_log = np.log(training.values)

#get variance stablized 
fig = plt.figure(figsize=(12,9))
ax1 = fig.add_subplot(211)
ax1.plot(training.values)
plt.title('No of Visitors')
ax2 = fig.add_subplot(212)
ax2.plot(training_log)
plt.title('Log of No of Visitors')

# Draw ACF and PACF
# For the log data
fig = plt.figure(figsize=(12,9))
ax1 = fig.add_subplot(211)
fig = smt.graphics.tsa.plot_acf(training_log, lags=40, ax=ax1)
ax1.set_title("ACF: Log Data")
ax2 = fig.add_subplot(212)
fig = smt.graphics.tsa.plot_pacf(training_log, lags=40, ax=ax2)
ax2.set_title("PACF: Log Data")

# get mean stablized
training_log_d = np.diff(training_log) 

# plot the transformed data
fig = plt.figure(figsize=(12,9))
ax1 = fig.add_subplot(211)
ax1.plot(training_log)
plt.title('Log of No of Visitors')
ax2 = fig.add_subplot(212)
ax2.plot(training_log_d)
plt.title('Differencing Log of No of Visitors')

# Check stationality again
# Plot ACF AND PACF for the ordinary difference of log data
# clear seasonality at 12 24 36 spike in ACF, NOT Stationary 
fig = plt.figure(figsize=(12,9))
ax1 = fig.add_subplot(211)
fig = smt.graphics.tsa.plot_acf(training_log_d, lags=40, ax=ax1)
ax1.set_title("ACF: first order difference of Log data")
ax2 = fig.add_subplot(212)
fig = smt.graphics.tsa.plot_pacf(training_log_d, lags=40, ax=ax2)
ax2.set_title("PACF: first order difference of Log data")

# data contains seasonal patterns, nonstationary
# Do seasonally differencing of log data
training_ds_log = training_log[12:] - training_log[:-12]
# Then first order difference
training_dsd_log = np.diff(training_ds_log);

# Plot seasonally differencing of log data
# Plot the first order difference on seasonally differenced log data
fig = plt.figure(figsize=(12,9))
ax1 = fig.add_subplot(211)
ax1.plot(training_ds_log)
plt.title('Seasonally Differenced log data')
ax2 = fig.add_subplot(212)
ax2.plot(training_dsd_log)
plt.title('Regular Difference of the Seasonal Differenced log data')

# Check stationality again
# Plot ACF and PACF for the seasonally differenced log data
fig = plt.figure(figsize=(12,9))
ax1 = fig.add_subplot(211)
fig = smt.graphics.tsa.plot_acf(training_ds_log, lags=40, ax=ax1)
ax1.set_title("ACF: seasonal difference of log data")
ax2 = fig.add_subplot(212)
fig = smt.graphics.tsa.plot_pacf(training_ds_log, lags=40, ax=ax2)  
ax2.set_title("PACF: seasonal difference of log data")   

# Check stationality again
# Plot ACF and PACF# For first order difference of seasonally differenced log data
# Stationary series 
fig = plt.figure(figsize=(12,9))
ax1 = fig.add_subplot(211)
fig = smt.graphics.tsa.plot_acf(training_dsd_log, lags=40, ax=ax1)
ax1.set_title("ACF: differencing the seasonally differenced Log data")
ax2 = fig.add_subplot(212)
fig = smt.graphics.tsa.plot_pacf(training_dsd_log, lags=40, ax=ax2)  
ax2.set_title("PACF: differencing the seasonally differenced Log data")   


#at seasonal level 
#ACF cutoff at lag 1 or 3(MA(Q=1 or 3)
#PACF cutoff at lag 2 (AR(P=2)
#D =1
#only choose one, either Seasonal MA or Seasonal AR

#at normal level for log data
#ACF dies down after lag 1 or 4 or 5 (MA(q=1 or 4 or 5))
#PACF dies down after lag 1，2，4 (AR(p=1 or2 or4))
#d=1

# Define the model according to the identified pattern
# 1 Sarima(1,1,1)(2,1,0)12
sarima_model_1= smt.tsa.statespace.SARIMAX(training_log, order=(1,1,1),seasonal_order=(2,1,0,12))   

# Estimating the model
result_1= sarima_model_1.fit(disp=False)
print(result_1.summary())

# 2 Sarima(1,1,4)(2,1,0)12
sarima_model_2= smt.tsa.statespace.SARIMAX(training_log, order=(1,1,4),seasonal_order=(2,1,0,12)) 
# Estimating the model
result_2= sarima_model_2.fit(disp=False)
print(result_2.summary())

# 3 Sarima(1,1,5)(2,1,0)12
sarima_model_3= smt.tsa.statespace.SARIMAX(training_log, order=(1,1,5),seasonal_order=(2,1,0,12)) 
# Estimating the model
result_3= sarima_model_3.fit(disp=False)
print(result_3.summary())

# 4 Sarima(2,1,4)(0,1,1)12
sarima_model_4= smt.tsa.statespace.SARIMAX(training_log, order=(2,1,4),seasonal_order=(0,1,1,12)) 
# Estimating the model
result_4= sarima_model_4.fit(disp=False)
print(result_4.summary())

# 5 Sarima(2,1,5)(0,1,1)12
sarima_model_5= smt.tsa.statespace.SARIMAX(training_log, order=(2,1,5),seasonal_order=(0,1,1,12)) 
# Estimating the model
result_5= sarima_model_5.fit(disp=False)
print(result_5.summary())

# 6 Sarima(4,1,5)(0,1,1)12
sarima_model_6= smt.tsa.statespace.SARIMAX(training_log, order=(4,1,5),seasonal_order=(0,1,1,12)) 
# Estimating the model
result_6= sarima_model_6.fit(disp=False)
print(result_6.summary())

# 7 Sarima(1,1,4)(0,1,3)12
sarima_model_7= smt.tsa.statespace.SARIMAX(training_log, order=(1,1,4),seasonal_order=(0,1,3,12)) 
# Estimating the model
result_7= sarima_model_7.fit(disp=False)
print(result_7.summary())

# based on AIC, model #2 Sarima(1,1,4)(2,1,0)12 is the best (AIC MIN)
# Forecasting
forecasts_sarima = result_2.forecast(27)
predicted_4 = np.exp(forecasts_sarima).tolist()

# Display forecasting on log data
fig = plt.figure(figsize=(12,6)) 
plt.plot(x_tr,training_log,label='log transformed data')
plt.plot(x_test,forecasts_sarima,label='forecast by SARIMA on log data')
plt.title('International Arrivals Forecast for Jan 2016 - Mar 2018 by SARIMA(1,1,4)(2,1,0)12' )
plt.legend(loc=2)

# Display forecasting on original data
fig = plt.figure(figsize=(12,6)) 
plt.plot(x_tr,training,label='observed')
plt.plot(x_test,predicted_4,label='forecast by SARIMA on original data')
plt.title('International Arrivals Forecast for Jan 2016 - Mar 2018 by SARIMA(1,1,4)(2,1,0)12' )
plt.legend(loc=2)

# Model accuracy 
# Evaluate the model performance   
def rmse (x,y):
    return np.sqrt(np.average(np.power(x-y,2)))

def mad (x,y):
    return np.average(np.abs(x-y))

def mape (x,y):
    return np.average(((np.abs(x-y))/x)*100)
#1
rmse_drift = rmse (test, predicted_1)
print ("the RMSE of Drift Method is {0}".format(rmse_drift))
mad_drift = mad (test, predicted_1)
print ("the MAD of Drift Method is {0}".format(mad_drift))
mape_drift = mape (test, predicted_1)
print ("the MAPE of Drift Method is {0}".format(mape_drift))
var_DriftMethod = np.var(test-predicted_1)

#2
rmse_SeasonalHoltsMultiplicative = rmse (test, predicted_2)
print ("the RMSE of Seasonal Holts Multiplicative is {0}".format(rmse_SeasonalHoltsMultiplicative))
mad_SeasonalHoltsMultiplicative = mad (test, predicted_2)
print ("the MAD of Seasonal Holts Multiplicative is {0}".format(mad_SeasonalHoltsMultiplicative))
mape_SeasonalHoltsMultiplicative= mape (test, predicted_2)
print ("the MAPE of Seasonal Holts Multiplicative is {0}".format(mape_SeasonalHoltsMultiplicative))
var_SeasonalHoltsMultiplicative = np.var(test- predicted_2)

#3 Decomposition Method 
rmse_DecompositionMultiplicative = rmse (test, predicted_3)
print ("the RMSE of Decomposition Method is {0}".format(rmse_DecompositionMultiplicative))
mad_DecompositionMultiplicative = mad (test, predicted_3)
print ("the MAD of Decomposition Method is {0}".format(mad_DecompositionMultiplicative))
mape_DecompositionMultiplicative = mape (test, predicted_3)
print ("the MAPE of Decomposition Method is {0}".format(mape_DecompositionMultiplicative))
var_DecompositionMethod = np.var(test-predicted_3)

#4
rmse_SARIMA = rmse (test, predicted_4)
print ("the RMSE of SARIMA is {0}".format(rmse_SARIMA))
mad_SARIMA = mad (test, predicted_4)
print ("the MAD of SARIMA is {0}".format(mad_SARIMA))
mape_SARIMA = mape (test, predicted_4)
print ("the MAPE of SARIMA is {0}".format(mape_SARIMA))
var_SARIMA = np.var(test- predicted_4)

# Based on aboved rmse,ranking is as below:
# #1 Sarima #2 Decomposition Method #3Seasonal Holts Multiplicative #4 Drift


# 5 combination of models
# combination will be based on only 3 models (benchmark, drift method will not be included)
# 5.1 equally weighted 
weight_equal = 1/3

forecast_combination_1 = weight_equal* (pd.Series(predicted_2)+pd.Series(predicted_3)+pd.Series(predicted_4))

predicted_5_1 = forecast_combination_1.tolist()

# 5.1 equally-weighted combination
rmse_forecast_combination_1 = rmse (test, predicted_5_1)
print ("the RMSE of equally-weighted combination is {0}".format(rmse_forecast_combination_1))
mad_forecast_combination_1 = mad (test, predicted_5_1)
print ("the MAD of equally-weighted combination is {0}".format(mad_forecast_combination_1))
mape_forecast_combination_1 = mape (test, predicted_5_1)
print ("the MAPE of equally-weighted combination is {0}".format(mape_forecast_combination_1))


# 5.2 optimal weights
#residuals for selected 2 models (rmse min:sarima and decomposition)
residual_sarima = test - predicted_4 
residual_decomposition = test - predicted_3

#covariance (will be a matrix)
covariance = np.cov(residual_sarima,residual_decomposition)

var_sarima = covariance[0,0]
var_decomposition = covariance[1,1]

r =  covariance [0,1] / (np.sqrt(var_sarima*var_decomposition))

#variance optimization weights
w_sarima = (var_decomposition - r*np.sqrt(var_sarima*var_decomposition)) / (var_sarima +var_decomposition-2*r*np.sqrt(var_sarima*var_decomposition))
w_decomposition = 1 - w_sarima

forecast_combination_2 = w_sarima * pd.Series(predicted_4)  + w_decomposition * pd.Series(predicted_3)
predicted_5_2 = forecast_combination_2.tolist()

# 5.2 optimal weight combination
rmse_forecast_combination_2 = rmse (test, predicted_5_2)
print ("the RMSE of optimal weight combination is {0}".format(rmse_forecast_combination_2))
mad_forecast_combination_2 = mad (test, predicted_5_2)
print ("the MAD of optimal weight combination is {0}".format(mad_forecast_combination_2))
mape_forecast_combination_2 = mape (test, predicted_5_2)
print ("the MAPE of optimal weight combination is {0}".format(mape_forecast_combination_2))

from statistics import variance
var_forecast_combination_2 = variance (test - predicted_5_2)

# 5.3 weights proportionally inverse to rmse 
sum_inversermse = 1/rmse_SeasonalHoltsMultiplicative + 1/rmse_DecompositionMultiplicative + 1/rmse_SARIMA
weight_SARIMA = (1/rmse_SARIMA)/sum_inversermse 
weight_DecompositionMultiplicative = (1/rmse_DecompositionMultiplicative)/sum_inversermse 
weight_SeasonalHoltsMultiplicative = (1/rmse_SeasonalHoltsMultiplicative )/sum_inversermse 

forecast_combination_3 = weight_SARIMA * pd.Series(predicted_4)  + weight_DecompositionMultiplicative * pd.Series(predicted_3)+ weight_SeasonalHoltsMultiplicative*pd.Series(predicted_2)
predicted_5_3 = forecast_combination_3.tolist()

# 5.3 weights proportionally inverse to rmse 
rmse_forecast_combination_3 = rmse (test, predicted_5_3)
print ("the RMSE of weights proportionally inverse to rmse combination is {0}".format(rmse_forecast_combination_3))
mad_forecast_combination_3 = mad (test, predicted_5_3)
print ("the MAD of weights proportionally inverse to rmse combination is {0}".format(mad_forecast_combination_3))
mape_forecast_combination_3 = mape (test, predicted_5_3)
print ("the MAPE of weights proportionally inverse to rmse combination is {0}".format(mape_forecast_combination_3))

# 5.4 weights inverse to their ranks
# rank1 sarima rank2 decomposition # rank3 holts
sum_rankinverse = 1 + 1/2 + 1/3
w_SARIMA = 1 / sum_rankinverse
w_DECOMPOSITION = (1/2) / sum_rankinverse
w_HOLTS = (1/3)/sum_rankinverse

forecast_combination_4 = w_SARIMA * pd.Series(predicted_4)  + w_DECOMPOSITION  * pd.Series(predicted_3)+ w_HOLTS*pd.Series(predicted_2)
predicted_5_4 = forecast_combination_4.tolist()

# 5.4 weights inverse to their ranks
rmse_forecast_combination_4 = rmse (test, predicted_5_4)
print ("the RMSE of weights inverse to their ranks is {0}".format(rmse_forecast_combination_4))
mad_forecast_combination_4 = mad (test, predicted_5_4)
print ("the MAD of weights inverse to their ranksis {0}".format(mad_forecast_combination_4))
mape_forecast_combination_4 = mape (test, predicted_5_4)
print ("the MAPE of weights inverse to their ranks is {0}".format(mape_forecast_combination_4))


# Based on RMSE, combination_2 is the best, followed by combination 3 combination 4 and combination 1.
# combination 2, it's actually the formula to optimize the portfolio variance, not rmse 
# combination 2 has higher rmse than sarima and higher variance than sarima and decomposition

# overall 
# sarima is the best 
# the forecast for the next year will be based on Sarima(1,1,4)(2,1,0)12







