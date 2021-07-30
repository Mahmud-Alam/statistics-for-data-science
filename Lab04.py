import pandas as pd

#%%

df = pd.read_csv('dataset_lab04.csv')
print(df.head())

#%%
# Problem - 01
# How many rows and columns this dataframe has?

rows = df.shape[0]
columns = df.shape[1]

print('\nNumber of rows:',rows,'\nNumber of columns:',columns)

#%%
# Problem - 02
# Describe (numerical summary) the time and amount column.

print(df.describe())

#%%
# Problem - 03
# There are 31 columns in the dataset. Compute some statistical measures like 
# mean, median, standard deviation, variance using Pandas Function.

print('\nMean:',round(df['Time'].mean(),5))
print('Median:',round(df['Time'].median(),5))
print('Standard Deviation:',round(df['Time'].std(),5))
print('variance:',round(df['Time'].var(),5))

print('\nMean:',round(df['Amount'].mean(),5))
print('Median:',round(df['Amount'].median(),5))
print('Standard Deviation:',round(df['Amount'].std(),5))
print('variance:',round(df['Amount'].var(),5))

#%%
# Problem - 04
# Compute the mean of any column using your own module and compare it with the mean value of Pandas.

import Lab04_MEAN_MODULE as mahmud

print('\nMy Own Mean Function:',round(mahmud.own_mean(df['Time']),5))
print('\nPandas Mean Function:',round(df['Time'].mean(),5))

#%%
# Problem - 05
# Show the histogram of Time and Amount column.

# Time Histogram
df.hist(column=['Time'], bins=50, color='lightgreen', edgecolor='black', linewidth=0.8)

# Amount Histogram
df.hist(column=['Amount'], bins=50, color='red', edgecolor='black', linewidth=0.8)

#%%
# Problem - 06
# Find the percentage of rows with class value = 0 (Non-Fraudulent) 
# and class value = 1 (Fraudulent).

## Count numbers
print('\nTotal count number: ')
print(df['Class'].value_counts())
print('\nPercentage of rows with class value 0 and 1:')

res = df['Class'].value_counts(normalize=True)*100
print(res)

print('\n(Non-Fraudulent = 0) : {}%'.format(100 - round(df['Class'].mean()*100, 2)))
print('(Fraudulent = 1)     : {}%'.format(round(df['Class'].mean()*100, 2)))

#%%
# Problem - 07
# Show the result you have got in 6 using a histogram.

r = pd.DataFrame(res)
print(r)
r.hist(column=['Class'],bins=10, color='red', edgecolor='black', linewidth=0.8)

#%%
# Problem - 8
# Show the histrogram (data distribution) of a few other columns. 
# Differentiate between left-skewed and right-skewed distributions.

# Right skewed distributions
print('\nRight skewed distributions:')

print('V4 Skeweness:',round(df['V4'].skew(),5))
df.hist(column=['V4'],bins=100, color='lightgreen', edgecolor='black', linewidth=0.5)

print('V4 Skeweness:',round(df['V6'].skew(),5))
df.hist(column=['V6'],bins=100, color='lightgreen', edgecolor='black', linewidth=0.5)

print('V4 Skeweness:',round(df['V7'].skew(),5))
df.hist(column=['V7'],bins=100, color='lightgreen', edgecolor='black', linewidth=0.5)

print('V4 Skeweness:',round(df['V9'].skew(),5))
df.hist(column=['V9'],bins=100, color='lightgreen', edgecolor='black', linewidth=0.5)

print('V4 Skeweness:',round(df['V10'].skew(),5))
df.hist(column=['V10'],bins=100, color='lightgreen', edgecolor='black', linewidth=0.5)


# Left skewed distributions
print('\nLeft skewed distributions:')

print('Standard Deviation:',round(df['V1'].skew(),5))
df.hist(column=['V1'],bins=100, color='darkred', edgecolor='black', linewidth=0.5)

print('Standard Deviation:',round(df['V2'].skew(),5))
df.hist(column=['V2'],bins=100, color='darkred', edgecolor='black', linewidth=0.5)

print('Standard Deviation:',round(df['V3'].skew(),5))
df.hist(column=['V3'],bins=100, color='darkred', edgecolor='black', linewidth=0.5)

print('Standard Deviation:',round(df['V5'].skew(),5))
df.hist(column=['V5'],bins=100, color='darkred', edgecolor='black', linewidth=0.5)

print('Standard Deviation:',round(df['V8'].skew(),5))
df.hist(column=['V8'],bins=100, color='darkred', edgecolor='black', linewidth=0.5)

#%%
# Problem - 9
# Find positive correlations among columns.

import seaborn as sns

corr_matrix = df.corr()

# Positive correlations amon columns
positive_corr_matrix = corr_matrix[corr_matrix >= 0]
print(positive_corr_matrix)

cmap = sns.color_palette('Greens')
sns.heatmap(positive_corr_matrix,cmap=cmap )


#%%
# Problem - 10
# Support your findings in Question 9 using a BoxPlot.

positive_corr_matrix.boxplot()

#%%
positive_corr_matrix.boxplot(column=['Time','Amount','Class'])

#%%
# problem -11
# Support your findings in Question 9 using a Scatter Plot.

positive_corr_matrix.plot.scatter(x='Time',y='Amount')

#%%
# Problem - 12
# Find negative correlations among columns.

corr_matrix = df.corr()

# Negative correlations amon columns
negative_corr_matrix = corr_matrix[corr_matrix < 0]
print(corr_matrix)

cmap = sns.color_palette('Reds')
sns.heatmap(negative_corr_matrix,cmap=cmap )

#%%
# problem -13
# Support your findings in Question 12 using a BoxPlot.

negative_corr_matrix.boxplot()

#%%
negative_corr_matrix.boxplot(column=['Time','Amount','Class'])

#%%
# problem -14
# Support your findings in Question 12 using a Scatter Plot.

negative_corr_matrix.plot.scatter(x='Time',y='Amount')
