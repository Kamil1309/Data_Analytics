import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Data1.csv") #1 Import data

#print(data.head()) #show first 5 rows of data

data.set_index('Unnamed: 0', inplace = True) #2 Set first column as the index

data.plot() #3 Plot all columns as time series.
plt.show() 


sns.set(style="whitegrid", color_codes=True) # set grid style
dfm = data.melt(var_name='column')
g = sns.FacetGrid(dfm, col='column', sharex= False) 
g.map(plt.hist, "value", bins = 10) # 4 Plot histograms of all columns
#g.map(sns.distplot, 'value', bins=10) # Plot histograms with Kernel Denisty Estimators on one chart
plt.show()

g = sns.FacetGrid(dfm, col='column', sharex= False) 
g.map(sns.kdeplot, 'value') # 5 Plot Kernel Denisty Estimators
plt.show()

##################### 6. analysis for columns 1-4


new_data = data.loc[data.index.str.startswith('2018'), "theta_1":"theta_4"] 
#new_data = data.loc["2018-01-01":"2018-12-31", "theta_1":"theta_4"] # another way

new_data.plot() #3 Plot all columns as time series.
plt.show() 

new_dfm = new_data.melt(var_name='column')
new_g = sns.FacetGrid(new_dfm, col='column', sharex= False) 
new_g.map(plt.hist, "value", bins = 10) # 4 Plot histograms of all columns
#new_g.map(sns.distplot, 'value', bins=10) # Plot histograms with Kernel Denisty Estimators on one chart
plt.show()

new_g = sns.FacetGrid(new_dfm, col='column', sharex= False) 
new_g.map(sns.kdeplot, 'value') # 5 Plot Kernel Denisty Estimators
plt.show()


