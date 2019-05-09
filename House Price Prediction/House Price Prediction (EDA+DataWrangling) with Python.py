
# coding: utf-8

# <a href="https://github.com/xiuwenbo?tab=repositories">
#     <img src="https://raw.githubusercontent.com/xiuwenbo/Markdown-Photos/master/IMG_5508.JPG" width="200" align="center">
# </a>  
# 
# 
#                                                                                                     @Xiuwenbo
#                                                                                                     
# <h1 align=center><font size="5"> House Price Prediction (EDA+DataWrangling) with Python </font></h1>

# ### Import some required libraries to begin with 

# In[1]:

import numpy as np # linear algebra
import pandas as pd # data processing


# In[2]:

from subprocess import check_output
print(check_output(["ls", "./data/"]).decode("utf8")) #check the files available in the directory


# ### About the dataset

# The dataset is about the sale of individual residential property in Ames, Iowa from 2006 to 2010. The data set contains 2930 observations and a large number of explanatory variables (23 nominal, 23 ordinal, 14 discrete, and 20 continuous) involved in assessing home values. 

# ### Data fields
# Here's a brief version of what you'll find in the data description file.
# 
# * SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
# * MSSubClass: The building class
# * MSZoning: The general zoning classification
# * LotFrontage: Linear feet of street connected to property
# * LotArea: Lot size in square feet
# * Street: Type of road access
# * Alley: Type of alley access
# * LotShape: General shape of property
# * LandContour: Flatness of the property
# * Utilities: Type of utilities available
# * LotConfig: Lot configuration
# * LandSlope: Slope of property
# * Neighborhood: Physical locations within Ames city limits
# * Condition1: Proximity to main road or railroad
# * Condition2: Proximity to main road or railroad (if a second is present)
# * BldgType: Type of dwelling
# * HouseStyle: Style of dwelling
# * OverallQual: Overall material and finish quality
# * OverallCond: Overall condition rating
# * YearBuilt: Original construction date
# * YearRemodAdd: Remodel date
# * RoofStyle: Type of roof
# * RoofMatl: Roof material
# * Exterior1st: Exterior covering on house
# * Exterior2nd: Exterior covering on house (if more than one material)
# * MasVnrType: Masonry veneer type
# * MasVnrArea: Masonry veneer area in square feet
# * ExterQual: Exterior material quality
# * ExterCond: Present condition of the material on the exterior
# * Foundation: Type of foundation
# * BsmtQual: Height of the basement
# * BsmtCond: General condition of the basement
# * BsmtExposure: Walkout or garden level basement walls
# * BsmtFinType1: Quality of basement finished area
# * BsmtFinSF1: Type 1 finished square feet
# * BsmtFinType2: Quality of second finished area (if present)
# * BsmtFinSF2: Type 2 finished square feet
# * BsmtUnfSF: Unfinished square feet of basement area
# * TotalBsmtSF: Total square feet of basement area
# * Heating: Type of heating
# * HeatingQC: Heating quality and condition
# * CentralAir: Central air conditioning
# * Electrical: Electrical system
# * 1stFlrSF: First Floor square feet
# * 2ndFlrSF: Second floor square feet
# * LowQualFinSF: Low quality finished square feet (all floors)
# * GrLivArea: Above grade (ground) living area square feet
# * BsmtFullBath: Basement full bathrooms
# * BsmtHalfBath: Basement half bathrooms
# * FullBath: Full bathrooms above grade
# * HalfBath: Half baths above grade
# * Bedroom: Number of bedrooms above basement level
# * Kitchen: Number of kitchens
# * KitchenQual: Kitchen quality
# * TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
# * Functional: Home functionality rating
# * Fireplaces: Number of fireplaces
# * FireplaceQu: Fireplace quality
# * GarageType: Garage location
# * GarageYrBlt: Year garage was built
# * GarageFinish: Interior finish of the garage
# * GarageCars: Size of garage in car capacity
# * GarageArea: Size of garage in square feet
# * GarageQual: Garage quality
# * GarageCond: Garage condition
# * PavedDrive: Paved driveway
# * WoodDeckSF: Wood deck area in square feet
# * OpenPorchSF: Open porch area in square feet
# * EnclosedPorch: Enclosed porch area in square feet
# * 3SsnPorch: Three season porch area in square feet
# * ScreenPorch: Screen porch area in square feet
# * PoolArea: Pool area in square feet
# * PoolQC: Pool quality
# * Fence: Fence quality
# * MiscFeature: Miscellaneous feature not covered in other categories
# * MiscVal: $Value of miscellaneous feature
# * MoSold: Month Sold
# * YrSold: Year Sold
# * SaleType: Type of sale
# * SaleCondition: Condition of sale

# ### Load Data from CSV File

# In[3]:

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')


# Let's limit floats output to 2 decimal points for convenience.

# In[4]:

pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x)) 


# ***

# ## Exploratory Data Analysis (**EDA**)

# Initially import some required libraries in this section.
# 
# Import the warnings library in case you are not a fan of being warned.

# In[4]:

import matplotlib.pyplot as plt  # Matlab-style plotting
get_ipython().magic('matplotlib inline')
import seaborn as sns
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from e.g. seaborn)


# I like the plot I draw has a clear style with a dark background

# In[6]:

sns.set_style('darkgrid')


# Let's have a overview of features in the data set and relations to the target variable first.  

# Apply **Pandas head() method, shape() method and info() method** to get a first overview of the train and test dataset and to answer how many rows and columns are there and what are the names of the features (columns).

# In[ ]:

#display the first five rows of the train dataset.
train.head(5)


# In[8]:

#display the first five rows of the test dataset.
test.head(5)


# check the numbers of samples and features.

# In[9]:

print('The training data size before dropping Id feature is : {} '.format(train.shape))
print('The test data size before dropping Id feature is : {} '.format(test.shape))


# Check the information of the data set

# In[10]:

print('The information of training data is:')      
print(train.info())


# In[11]:

print('The information of test data is:')      
print(test.info())


# **Pandas describe() method** gives a summary of the statistics (only for numerical columns)

# In[12]:

print('The summary of training data is:')      
print(train.describe())


# In[13]:

print('The summary of test data is:')      
print(test.describe())


# In[ ]:




# ### **Target variable**
# 
# In this project, we train the data to make various moedls to predict the House Price, aka the column 'SalePrice' in the data set. Therefore, we regard 'SalePrice' as the targer variable.
# 
# Let's import the relevant libraries firstly and have a look at its normal distribution.

# In[14]:

from scipy import stats
from scipy.stats import norm, skew #for some statistics


# In[15]:

# Get the fitted parameters used by the function and limit floats output to 2 decimal points.
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))


# In[16]:

sns.distplot(train['SalePrice'] , fit=norm)
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')


# Let's plot Q-Q plot to compare the shapes of distributions, providing a graphical view of how properties such as location, scale, and skewness are similar or different in the two distributions. Q–Q plots can be used to compare collections of data, or theoretical distributions. 

# In[17]:

fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)


# If one tail is longer than another, the distribution is skewed.
# * A left-skewed distribution has a long left tail. Left-skewed distributions are also called negatively-skewed distributions. That’s because there is a long tail in the negative direction on the number line. The mean is also to the left of the peak.
# * A right-skewed distribution has a long right tail. Right-skewed distributions are also called positive-skew distributions. That’s because there is a long tail in the positive direction on the number line. The mean is also to the right of the peak.

# The target variable is right skewed. As (linear) models love normally distributed data , we need to transform this variable and make it more normally distributed.

# The log transformation is the most popular among the different types of transformations used to transform skewed data to approximately conform to normality. If the original data follows a log-normal distribution or approximately so, then the log-transformed data follows a normal or near normal distribution.

# Let's apply the numpy fuction log1p which use log(1+x) to all elements of the column

# In[18]:

train["SalePrice"] = np.log1p(train["SalePrice"])


# In[19]:

# Get the fitted parameters used by the function and plot the distribution 
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
sns.distplot(train['SalePrice'] , fit=norm)


# Let's replot Q-Q plot to compare the shapes of distributions

# In[20]:

fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)


# This is much better. 

# ***

# ### **Check the missing data**

# In[21]:

# check the missing value in training data
missing_train = train.isnull().sum().sort_values(ascending = False)
missing_train


# In[22]:

#plot the top 10 missing values
missing_x_axis = missing_train[:10]
missing_y_axis = missing_train[:10].index
width = 10
height = 8
plt.figure(figsize=(width, height))

sns.barplot(missing_x_axis, missing_y_axis)
plt.title('Missing value in trianing data')


# In[23]:

# check the top 10 missing values in test data
missing_test = test.isnull().sum().sort_values(ascending = False)

missing_x_axis = missing_test[:10]
missing_y_axis = missing_test[:10].index

width = 10
height = 8
plt.figure(figsize=(width, height))

sns.barplot(missing_x_axis, missing_y_axis)
plt.title('Missing value in test data')


# The plot above illustrate that there are plenty of variables are missing value. 
# 
# Before we proceed to deeper analysis, we need to have a look at these missing variables during the process of EDA. After that, I am going to apply feature engineering to deal with those missing value and make more correlated variables to make a accurate prediction model.

# ### **Filling the missing values**
# 
# For a few columns there is lots of NaN entries. However, reading the data description we find this is not missing data: For example, PoolQC, NaN is not missing data but means no pool, likewise for Fence, FireplaceQu etc.

# In[24]:

# columns where NaN values have meaning e.g. no pool etc.
cols_fillna = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu',
               'GarageQual','GarageCond','GarageFinish','GarageType', 'Electrical',
               'KitchenQual', 'SaleType', 'Functional', 'Exterior2nd', 'Exterior1st',
               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2',
               'MSZoning', 'Utilities']

# replace 'NaN' with 'None' in these columns
for col in cols_fillna:
    train[col].fillna('None',inplace=True)
    test[col].fillna('None',inplace=True)


# In[25]:

missing_total = train.isnull().sum().sort_values(ascending=False)
missing_percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([missing_total, missing_percent], axis=1, keys=['Missing Value Total', 'Percent'])
missing_data.head()


# In[26]:

# fillna with mean for the remaining columns: LotFrontage, GarageYrBlt, MasVnrArea
cols_fillna = ['LotFrontage', 'GarageYrBlt', 'MasVnrArea']

for col in cols_fillna:
    train[col].fillna(train[col].mean(), inplace=True)
    test[col].fillna(test[col].mean(), inplace=True)


# In[27]:

missing_total = train.isnull().sum().sort_values(ascending=False)
missing_percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([missing_total, missing_percent], axis=1, keys=['Missing Values Total', 'Percent'])
missing_data.head()


# ***

# Like the target variable, some of the feature values are not normally distributed and it is therefore better to use log values in both training and test data. 
# 
# Initially let's divide the feature into numerical and categorical and find out some interesting details. 

# In[28]:

numerical_feats = train.dtypes[train.dtypes != "object"].index
print("Number of Numerical features: ", len(numerical_feats))

categorical_feats = train.dtypes[train.dtypes == "object"].index
print("Number of Categorical features: ", len(categorical_feats))


# In[29]:

print(train[numerical_feats].columns)
print("*"*100)
print(train[categorical_feats].columns)


# ### skewness and kurtosis
# Let's check for skewness and kurtosis in numerical features

# In[30]:

for col in numerical_feats:
    print('{:15}'.format(col), 
          'Skewness: {:05.2f}'.format(train[col].skew()) , 
          '   ' ,
          'Kurtosis: {:06.2f}'.format(train[col].kurt())  
         )


# In[31]:

skewed_features = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF2', 'LowQualFinSF', 'GrLivArea'
                   , 'BsmtHalfBath', 'BsmtFinSF1', 'TotalBsmtSF', 'WoodDeckSF', 'OpenPorchSF'
                   , 'KitchenAbvGr', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']


# In[32]:

for feature in skewed_features:
    train[feature] = np.log1p(train[feature])
    test[feature] = np.log1p(test[feature])


# In[33]:

len(numerical_feats)


# In[34]:

nr_rows = 12
nr_cols = 3

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3))

li_num_feats = list(numerical_feats)
li_not_plot = ['Id', 'SalePrice']
li_plot_num_feats = [c for c in list(numerical_feats) if c not in li_not_plot]


for r in range(0,nr_rows):
    for c in range(0,nr_cols):  
        i = r*nr_cols+c
        if i < len(li_plot_num_feats):
            sns.regplot(train[li_plot_num_feats[i]], train['SalePrice'], ax = axs[r][c])
            stp = stats.pearsonr(train[li_plot_num_feats[i]], train['SalePrice'])
            #axs[r][c].text(0.4,0.9,"title",fontsize=7)
            str_title = "r = " + "{0:.2f}".format(stp[0]) + "      " "p = " + "{0:.2f}".format(stp[1])
            axs[r][c].set_title(str_title,fontsize=11)
            
plt.tight_layout()    
plt.show()   


# ### Summary on numerical features:
# 
# * some of the features like "OverallQual" have strong linear correlation 82% towards the target. 
# * while some of other features like "MSSubClass" have a quite weak correlation to the taget variable. 
# * there are several features in the numerical way turns out to be the categorical based on the plots (like "OverallQual"). 

# ### Remove the outliers

# In[38]:

train = train.drop(
    train[(train['OverallQual']==10) & (train['SalePrice']<12.3)].index)
train = train.drop(
    train[(train['GrLivArea']>8.3) & (train['SalePrice']<12.5)].index)


# Find out the features which have a strong correlation to the target variable. 

# In[42]:

corr = train.corr()
corr_abs = corr.abs()
min_val_corr = 0.4    


nr_num_cols = len(numerical_feats)
ser_corr = corr_abs.nlargest(nr_num_cols, 'SalePrice')['SalePrice']

cols_abv_corr_limit = list(ser_corr[ser_corr.values > min_val_corr].index)
cols_bel_corr_limit = list(ser_corr[ser_corr.values <= min_val_corr].index)


# List the features and their correlation coeffient to the target variable.

# In[43]:

print(ser_corr)


# In[44]:

print("List of numerical features with r above min_val_corr :", cols_abv_corr_limit)


# In[45]:

print("List of numerical features with r below min_val_corr :", cols_bel_corr_limit)


# ***

# ### Let's turn to the categoical features.

# The unique value in these categorical features.

# In[47]:

for catg in list(categorical_feats) :
    print(train[catg].value_counts())
    print('*'*50)


# Their correlations to the target variable.

# In[50]:

nr_rows = 15
nr_cols = 3

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4,nr_rows*3))

for r in range(0,nr_rows):
    for c in range(0,nr_cols):  
        i = r*nr_cols+c
        if i < len(list(categorical_feats)):
            sns.boxplot(x=list(categorical_feats)[i], y='SalePrice', data=train, ax = axs[r][c])
    
plt.tight_layout()    
plt.show()   


# In[52]:

catg_strong_corr = [ 'MSZoning', 'Neighborhood', 'Condition2', 'MasVnrType', 'ExterQual', 
                     'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType']

catg_weak_corr = ['Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
                  'LandSlope', 'Condition1',  'BldgType', 'HouseStyle', 'RoofStyle', 
                  'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterCond', 'Foundation', 
                  'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 
                  'HeatingQC', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 
                  'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 
                  'SaleCondition']


# ### Confusion Matrix

# In[51]:

def plot_corr_matrix(df, nr_c, targ) :
    
    corr = df.corr()
    corr_abs = corr.abs()
    cols = corr_abs.nlargest(nr_c, targ)[targ].index
    cm = np.corrcoef(df[cols].values.T)

    plt.figure(figsize=(nr_c/1.5, nr_c/1.5))
    sns.set(font_scale=1.25)
    sns.heatmap(cm, linewidths=1.5, annot=True, square=True, 
                fmt='.2f', annot_kws={'size': 10}, 
                yticklabels=cols.values, xticklabels=cols.values
               )
    plt.show()


# In[53]:

nr_feats = len(cols_abv_corr_limit)


# In[56]:

plot_corr_matrix(train, nr_feats, 'SalePrice')


# ***

# ### Data Wrangling 

# In this section, the priority is to drop the less correlated features to the target variable in the dataset. Plus, transform some of the catregorical features to the numerical ones.
# 
# In a nutshell: 
# 
# * for numerical features: drop the similar and less correlated features
# * for categorical features: transform them to numerical

# In[58]:

id_test = test['Id']

to_drop_num  = cols_bel_corr_limit
to_drop_catg = catg_weak_corr

cols_to_drop = ['Id'] + to_drop_num + to_drop_catg 

for df in [train, test]:
    df.drop(cols_to_drop, inplace= True, axis = 1)


# In[64]:

catg_list = catg_strong_corr.copy()
catg_list.remove('Neighborhood')

for catg in catg_list :
    sns.boxenplot(x=catg, y='SalePrice', data=train)
    plt.show()


# In[66]:

fig, ax = plt.subplots()
fig.set_size_inches(16, 5)
sns.boxenplot(x='Neighborhood', y='SalePrice', data=train, ax=ax)
plt.xticks(rotation=45)
plt.show()


# In[68]:

for catg in catg_list :
    group = train.groupby(catg)['SalePrice'].mean()
    print(group)


# In[69]:

# 'MSZoning'
msz_catg2 = ['RM', 'RH']
msz_catg3 = ['RL', 'FV'] 


# Neighborhood
nbhd_catg2 = ['Blmngtn', 'ClearCr', 'CollgCr', 'Crawfor', 'Gilbert', 'NWAmes', 'Somerst', 'Timber', 'Veenker']
nbhd_catg3 = ['NoRidge', 'NridgHt', 'StoneBr']

# Condition2
cond2_catg2 = ['Norm', 'RRAe']
cond2_catg3 = ['PosA', 'PosN'] 

# SaleType
SlTy_catg1 = ['Oth']
SlTy_catg3 = ['CWD']
SlTy_catg4 = ['New', 'Con']


# In[71]:

for df in [train, test]:
    
    df['MSZ_num'] = 1  
    df.loc[(df['MSZoning'].isin(msz_catg2) ), 'MSZ_num'] = 2    
    df.loc[(df['MSZoning'].isin(msz_catg3) ), 'MSZ_num'] = 3        
    
    df['NbHd_num'] = 1       
    df.loc[(df['Neighborhood'].isin(nbhd_catg2) ), 'NbHd_num'] = 2    
    df.loc[(df['Neighborhood'].isin(nbhd_catg3) ), 'NbHd_num'] = 3    

    df['Cond2_num'] = 1       
    df.loc[(df['Condition2'].isin(cond2_catg2) ), 'Cond2_num'] = 2    
    df.loc[(df['Condition2'].isin(cond2_catg3) ), 'Cond2_num'] = 3    
    
    df['Mas_num'] = 1       
    df.loc[(df['MasVnrType'] == 'Stone' ), 'Mas_num'] = 2 
    
    df['ExtQ_num'] = 1       
    df.loc[(df['ExterQual'] == 'TA' ), 'ExtQ_num'] = 2     
    df.loc[(df['ExterQual'] == 'Gd' ), 'ExtQ_num'] = 3     
    df.loc[(df['ExterQual'] == 'Ex' ), 'ExtQ_num'] = 4     
   
    df['BsQ_num'] = 1          
    df.loc[(df['BsmtQual'] == 'Gd' ), 'BsQ_num'] = 2     
    df.loc[(df['BsmtQual'] == 'Ex' ), 'BsQ_num'] = 3     
    
    df['CA_num'] = 0          
    df.loc[(df['CentralAir'] == 'Y' ), 'CA_num'] = 1    

    df['Elc_num'] = 1       
    df.loc[(df['Electrical'] == 'SBrkr' ), 'Elc_num'] = 2 


    df['KiQ_num'] = 1       
    df.loc[(df['KitchenQual'] == 'TA' ), 'KiQ_num'] = 2     
    df.loc[(df['KitchenQual'] == 'Gd' ), 'KiQ_num'] = 3     
    df.loc[(df['KitchenQual'] == 'Ex' ), 'KiQ_num'] = 4      
    
    df['SlTy_num'] = 2       
    df.loc[(df['SaleType'].isin(SlTy_catg1) ), 'SlTy_num'] = 1  
    df.loc[(df['SaleType'].isin(SlTy_catg3) ), 'SlTy_num'] = 3  
    df.loc[(df['SaleType'].isin(SlTy_catg4) ), 'SlTy_num'] = 4  


# In[72]:

new_col_num = ['MSZ_num', 'NbHd_num', 'Cond2_num'
               , 'Mas_num', 'ExtQ_num', 'BsQ_num'
               , 'CA_num', 'Elc_num', 'KiQ_num', 'SlTy_num']


# In[75]:

nr_rows = 4
nr_cols = 3

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3))

for r in range(0,nr_rows):
    for c in range(0,nr_cols):  
        i = r*nr_cols+c
        if i < len(new_col_num):
            sns.regplot(train[new_col_num[i]], train['SalePrice'], ax = axs[r][c])
            stp = stats.pearsonr(train[new_col_num[i]], train['SalePrice'])
            str_title = "r = " + "{0:.2f}".format(stp[0]) + "      " "p = " + "{0:.2f}".format(stp[1])
            axs[r][c].set_title(str_title,fontsize=11)
            
plt.tight_layout()    


# In these plots, we can see there are several features like "NbHd_num, ExtQ_num, BsQ_num, KiQ_num" have a strong correlation to the target variable.

# In[76]:

catg_cols_to_drop = ['Neighborhood' , 'Condition2', 'MasVnrType'
                     , 'ExterQual', 'BsmtQual','CentralAir', 'Electrical'
                     , 'KitchenQual', 'SaleType']


# In[78]:

corr1 = train.corr()
corr_abs_1 = corr1.abs()

nr_all_cols = len(train)
ser_corr_1 = corr_abs_1.nlargest(nr_all_cols, 'SalePrice')['SalePrice']

print(ser_corr_1)


# In[80]:

cols_bel_corr_limit_1 = list(ser_corr_1[ser_corr_1.values <= min_val_corr].index)
for df in [train, test] :
    df.drop(catg_cols_to_drop, inplace= True, axis = 1)
    df.drop(cols_bel_corr_limit_1, inplace= True, axis = 1)    


# Finally, let's list of all features with strong correlation to target variable.

# In[83]:

corr = train.corr()
corr_abs = corr.abs()

nr_all_cols = len(train)
print (corr_abs.nlargest(nr_all_cols, 'SalePrice')['SalePrice'])


# In[84]:

nr_feats=len(train.columns)
plot_corr_matrix(train, nr_feats, 'SalePrice')


# ### Multicollinearity
# 
# Multicollinearity (or inter correlation) exists when at least some of the predictor variables are correlated among themselves.
# 
# Strong correlation of these features to other, similar features:
# 
# * 'GrLivArea_Log' and 'TotRmsAbvGrd'
# 
# * 'GarageCars' and 'GarageArea'
# 
# * 'TotalBsmtSF' and '1stFlrSF'
# 
# * 'YearBuilt' and 'GarageYrBlt'
# 
# Of those features we drop the one  with a less correlated coeffiecient to target variable.

# In[86]:

# switch for dropping columns that are similar to others already used and show a high correlation to these     
drop_similar = 1


# In[87]:

cols = corr_abs.nlargest(nr_all_cols, 'SalePrice')['SalePrice'].index
cols = list(cols)

if drop_similar == 1 :
    for col in ['GarageArea','1stFlrSF','TotRmsAbvGrd','GarageYrBlt'] :
        if col in cols: 
            cols.remove(col)


# In[88]:

print(list(cols))


# Get the list of the features/columns for the following modelling

# In[89]:

feats = cols.copy()
feats.remove('SalePrice')

print(feats)


# In[91]:

df_train_ml = train[feats].copy()
df_test_ml  = test[feats].copy()

y = train['SalePrice']


# In[93]:

all_data = pd.concat((train[feats], test[feats]))


# In[96]:

df_train_ml = all_data[:train.shape[0]]
df_test_ml  = all_data[train.shape[0]:]


# ### StandardScaler 
# 
# Standardize features by removing the mean and scaling to unit variance

# In[97]:

from sklearn.preprocessing import StandardScaler


# In[98]:

sc = StandardScaler()
df_train_ml_sc = sc.fit_transform(df_train_ml)
df_test_ml_sc = sc.transform(df_test_ml)


# In[100]:

df_train_ml_sc = pd.DataFrame(df_train_ml_sc)
df_test_ml_sc = pd.DataFrame(df_test_ml_sc)


# ### Creating the new dataset for the further modelling

# In[102]:

X = df_train_ml.copy()
y = train['SalePrice']
X_test = df_test_ml.copy()

X_sc = df_train_ml_sc.copy()
y_sc = train['SalePrice']
X_test_sc = df_test_ml_sc.copy()


# In[103]:

X.info()


# In[104]:

X_test.info()


# ***
