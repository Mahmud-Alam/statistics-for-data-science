import numpy as np
import pandas as pd
from sklearn import preprocessing

#%%

# Read CSV File

df_actual = pd.read_csv('student_satisfaction.csv')
print(df_actual.head())
print(df_actual.shape)

#%%

# all columns into a array

columns = np.array(df_actual.columns)
print(columns)

#%%

# Copy df_actual into df

df = df_actual.copy()
print(df)

#%%

# Drop all unnecessary columns

#df.drop(['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12',],axis=1, inplace=True)
df.drop(df.iloc[:,:12],axis=1, inplace=True)

df.drop(['Satisfied'],axis=1, inplace=True)

print(df)

#%%

# Check whether there is any missing value or not

print(df.isnull().sum())

#%%

# Replace missing values with median

df['IA1'].fillna(df['IA1'].median(skipna=True), inplace=True)
df['IA3'].fillna(df['IA3'].median(skipna=True), inplace=True)
df['TLA2'].fillna(df['TLA2'].median(skipna=True), inplace=True)

#%%

# Check again for missing values 

print(df.isnull().sum())

#%%

# Check the shape of the dataset

print(df.shape)

#%%

# DataFrame1 - df1

# Encoding via Get-Dummies technique

df_IA1 = pd.get_dummies(df['IA1'],prefix='IA1')
df_IA2 = pd.get_dummies(df['IA2'],prefix='IA2')
df_IA3 = pd.get_dummies(df['IA3'],prefix='IA3')
df_IA4 = pd.get_dummies(df['IA4'],prefix='IA4')
df_IA5 = pd.get_dummies(df['IA5'],prefix='IA5')
df_IA6 = pd.get_dummies(df['IA6'],prefix='IA6')

df1 = pd.concat([ df,df_IA1,df_IA2,df_IA3,df_IA4,df_IA5,df_IA6],axis=1)

print(df1.head())
print(df1.shape)

df1.drop(df1.iloc[:,:6],axis=1, inplace=True)

print(df1.head())
print(df1.shape)

#%%

df_AA1 = pd.get_dummies(df['AA1'],prefix='AA1')
df_AA2 = pd.get_dummies(df['AA2'],prefix='AA2')
df_AA3 = pd.get_dummies(df['AA3'],prefix='AA3')
df_AA4 = pd.get_dummies(df['AA4'],prefix='AA4')
df_AA5 = pd.get_dummies(df['AA5'],prefix='AA5')
df_AA6 = pd.get_dummies(df['AA6'],prefix='AA6')

df1 = pd.concat([ df1,df_AA1,df_AA2,df_AA3,df_AA4,df_AA5,df_AA6],axis=1)

print(df1.head())
print(df1.shape)

df1.drop(df1.iloc[:,:6],axis=1, inplace=True)

print(df1.head())
print(df1.shape)

#%%

df_TLA1 = pd.get_dummies(df['TLA1'],prefix='TLA1')
df_TLA2 = pd.get_dummies(df['TLA2'],prefix='TLA2')
df_TLA3 = pd.get_dummies(df['TLA3'],prefix='TLA3')
df_TLA4 = pd.get_dummies(df['TLA4'],prefix='TLA4')
df_TLA5 = pd.get_dummies(df['TLA5'],prefix='TLA5')
df_TLA6 = pd.get_dummies(df['TLA6'],prefix='TLA6')
df_TLA7 = pd.get_dummies(df['TLA7'],prefix='TLA7')

df1 = pd.concat([ df1,df_TLA1,df_TLA2,df_TLA3,df_TLA4,df_TLA5,df_TLA6,df_TLA7],axis=1)

print(df1.head())
print(df1.shape)

df1.drop(df1.iloc[:,:7],axis=1, inplace=True)

print(df1.head())
print(df1.shape)

#%%

# Replace OR Column

df_OR = df[['OR']]

df1.drop(['OR'],axis=1, inplace=True)

df1 = pd.concat([df1,df_OR],axis=1)

#%%

# DataFrame1 - df1

print(df1.head())
print(df1.shape)

#%%

# DataFrame2 - df2

df2 = df.copy()
print(df2.head())
print(df2.shape)

