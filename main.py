import pandas as pd
import numpy as np
import Group1 as g1

#%%

### Numeric Data

df = pd.read_csv('german-data-numeric.csv',header=None)
df.columns = ['Status of existing checking account','Duration in month','Credit history','Purpose','Credit amount','Savings account/bonds','Present employment since','Installment rate in percentage of disposable income','Personal status and sex','Other debtors / guarantors','Present residence since','Property','Age in years','Other installment plans','Housing','Number of existing credits at this bank','Job','Number of people being liable to provide maintenance for','Telephone','foreign worker','Cost Matrix','Column-22','Column-23','Column-24','Column-25',]

print(df.info())

#%%

series1 = df['Duration in month']
print(series1.head())
series2 = df['Purpose']
print(series2.head())
series3 = df['Other debtors / guarantors']
print(series3.head())

#%%

# 1
# count Function
print ("\nModule count output: ",g1.COUNT(series1))
print('Pandas count Output: ',series1.count())

#%%

# 2
# Describe Function
print ("\nModule describe output: ",g1.DESCRIBE(series1))
print('Pandas describe Output: ',series1.describe())

#%%

# 3
# Min Function
print('\nModule Min Output: ',g1.MIN(series1))
print('Pandas Min Output: ',series1.min())

#%%

# 4
# Min index Function
print('\nModule index location of Min Output: ',g1.ARGMIN(series1))
print('Pandas index location of Min Output: ',series1.argmin())

#%%

# 5
# Max Function
print('\nModule Max Output: ',g1.MAX(series1))
print('Pandas Max Output: ',series1.max())

#%%

# 6
# Max index Function
print('\nModule index location of Max Output: ',g1.ARGMAX(series1))
print('Pandas index location of Max Output: ',series1.argmax())

#%%

# 7
#index lebel at which minimum lies
print("\nModule minimum value's index lebel Output: ",g1.IDX_MIN(series1))
print("Pandas minimum value's index lebel Output: ",series1.idxmin())

#%%

# 8
#index lebel at which maximum lies
print("\nModule maximum value's index lebel Output: ",g1.IDX_MAX(series1))
print("Pandas maximum value's index lebel Output: ",series1.idxmax())

#%%

# 9
#QUANTILE
print("Module Output:", "%.1f" % g1.Quantile(series2, 0.5))
print("Pandas Output:", df[['Purpose']].quantile())

#%%

# 10
#SUM
print("Module Output:", g1.Sum(series2))
print("Pandas Output:", df.loc[:,'Purpose'].sum())

#%%

# 11
#MEAN
print("Module Output:", "%.6f" % g1.Mean(series2))
print("Pandas Output:", df[['Purpose']].mean())

#%%

# 12
#MEDIAN
print("Module Output:", "%.1f" % g1.Median(series2))
print("Pandas Output:", df[['Purpose']].median())

#%%

# 13
#MAD
print("Module Output:", "%.6f" % g1.Mad(series2))
print("Pandas Output:", df[['Purpose']].mad())

#%%
# 14
#PROD
print("Module Output:", g1.Prod(series2))
print("Pandas Output:", df.loc[:,'Purpose'].prod())

#%%

# 15
#VAR
print("Module Output:", "%.6f" % g1.Var(series3))
print("Pandas Output:", df[['Other debtors / guarantors']].var())

#%%

# 16
#STD
print("Module Output:", "%.16f" % g1.Std(series3))
print("Pandas Output:", df.loc[:,'Other debtors / guarantors'].std())

#%%

# 17
# Skewness Function
print('\nModule Skewness Output: ',g1.SKEW(series1))
print('Pandas Skewness Output: ',series1.skew())

#%%

# 18
# Kurtosis Function
print('\nModule Kurtosis Output: ',g1.KURT(series1))
print('Pandas Kurtosis Output: ',series1.kurt())

#%%

# 19
# CumSum Function
print('\nModule CumSum Output: ',g1.CUMSUM(series1))
print('\nPandas CumSum Output: ',series1.cumsum())

#%%

# 20
# CumMin Function
print('\nModule CumMin Output: ',g1.CUMMIN(series1))
print('\nPandas CumMin Output: ',series1.cummin())

#%%

# 21
# CumMax Function
print('\nModule CumMax Output: ',g1.CUMMAX(series1))
print('\nPandas CumMax Output: ',series1.cummax())

#%%

# 22
# CumProd Function
print('\nModule CumProd Output: ',g1.CUMPROD(series1))
print('\nPandas CumProd Output: ',series1.cumprod())

#%%

# 23
# Diff Function
print('\nModule Diff Output: ',g1.DIFF(series1))
print('\nPandas Diff Output: ',series1.diff())

#%%

# 24
# pct_change Function
print('\nModule pct_change Output: ',g1.PCT_CHANGE(series1))
print('\nPandas pct_change Output: ',series1.pct_change())

#%%

# 25
# Trimmed-Mean Function
print('\nModule Trimmed-Mean Output: ',g1.TRIMMED_MEAN(series1, 100))

#%%

# 26
# Weighted-Mean Function
wt = np.random.randint(1,5, size=1000)
print('\nModule Weighted-Mean Output: ',g1.WEIGHTED_MEAN(series1, wt))

#%%

# 27
# Weighted-Median Function
wt = np.random.randint(1,5, size=1000)
print('\nModule Weighted-Median Output: ',g1.WEIGHTED_MEDIAN(series1, wt))

#%%

# 28
# Mode Function
print('\nModule Mode Output: ',g1.MODE(series1))
print('Pandas Mode Output: ',series1.mode())

#%%

# 29
# Dispersion Function
print('\nModule Dispersion Output: ',g1.DISPERSION(series1))

#%%

# 30
# Z-Score Function
print('\nModule Z-Score Output: ',g1.ZSCORE(series1))

#%%

# 31
# Covariance Function
print('\nModule Covariance Output: ',g1.COVARIANCE(series1, series2))

#%%

# 32
# Correlation Function
print('\nModule Correlation Output: ',g1.CORRELATION(series1, series2))

#%%

# 33
#INTERQUANTILE RANGE
print("Module Output:", g1.INTERQUARTILE_RANGE(series3))

#%%

# 34
#Standard Error
print("Module Output:", g1.StandardError(series2))

#%%

# 35
#Confidence Interval
print("Module Output:", g1.ConfidenceInterval(df.loc[1,'Purpose'],series2))

