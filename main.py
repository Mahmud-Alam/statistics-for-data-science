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

#%%

# Min Function
print('\nModule Min Output: ',g1.MIN(series1))
print('Pandas Min Output: ',series1.min())

#%%

# Max Function
print('\nModule Max Output: ',g1.MAX(series1))
print('Pandas Max Output: ',series1.max())

#%%

# Sum Function
print('\nModule Sum Output: ',g1.SUM(series1))
print('Pandas Sum Output: ',series1.sum())

#%%

# Mean Function
print('\nModule Mean Output: ',g1.MEAN(series1))
print('Pandas Mean Output: ',series1.mean())

#%%

# Variance Function
print('\nModule Variance Output: ',g1.VAR(series1))
print('Pandas Variance Output: ',series1.var())

#%%

# STD Function
print('\nModule STD Output: ',g1.STD(series1))
print('Pandas std Output: ',series1.std())

#%%

# Skewness Function
print('\nModule Skewness Output: ',g1.SKEW(series1))
print('Pandas Skewness Output: ',series1.skew())

#%%

# Kurtosis Function
print('\nModule Kurtosis Output: ',g1.KURT(series1))
print('Pandas Kurtosis Output: ',series1.kurt())

#%%

# CumSum Function
print('\nModule CumSum Output: ',g1.CUMSUM(series1))
print('\nPandas CumSum Output: ',series1.cumsum())

#%%

# CumMin Function
print('\nModule CumMin Output: ',g1.CUMMIN(series1))
print('\nPandas CumMin Output: ',series1.cummin())

#%%

# CumMax Function
print('\nModule CumMax Output: ',g1.CUMMAX(series1))
print('\nPandas CumMax Output: ',series1.cummax())

#%%

# CumProd Function
print('\nModule CumProd Output: ',g1.CUMPROD(series1))
print('\nPandas CumProd Output: ',series1.cumprod())

#%%

# Diff Function
print('\nModule Diff Output: ',g1.DIFF(series1))
print('\nPandas Diff Output: ',series1.diff())

#%%

# pct_change Function
print('\nModule pct_change Output: ',g1.PCT_CHANGE(series1))
print('\nPandas pct_change Output: ',series1.pct_change())

#%%

# Mean-Absolute-Deviation Function
print('\nModule Mean-Absolute-Deviation Output: ',g1.MEAN_ABSOLUTE_DEVIATION(series1))

#%%

# Trimmed-Mean Function
print('\nModule Trimmed-Mean Output: ',g1.TRIMMED_MEAN(series1, 100))

#%%

# Weighted-Mean Function

wt = np.random.randint(1,5, size=1000)
print('\nModule Weighted-Mean Output: ',g1.WEIGHTED_MEAN(series1, wt))

#%%

# Weighted-Median Function
wt = np.random.randint(1,5, size=1000)
print('\nModule Weighted-Median Output: ',g1.WEIGHTED_MEDIAN(series1, wt))

#%%

# Mode Function
print('\nModule Mode Output: ',g1.MODE(series1))
print('Pandas Mode Output: ',series1.mode())

#%%

# Dispersion Function
print('\nModule Dispersion Output: ',g1.DISPERSION(series1))

#%%

# Z-Score Function
print('\nModule Z-Score Output: ',g1.ZSCORE(series1))

#%%

# Covariance Function
print('\nModule Covariance Output: ',g1.COVARIANCE(series1, series2))

#%%

# Correlation Function
print('\nModule Correlation Output: ',g1.CORRELATION(series1, series2))

