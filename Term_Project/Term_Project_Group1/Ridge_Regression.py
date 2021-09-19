import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

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


#%%

# =============================================================================
#  Dataset-1
#  Independent variable put into X and Dependent variable 'OR' put into y
# =============================================================================

X1 = df1.iloc[:,:-1]
print(X1)

y1 = df1[['OR']]
print(y1)

#%%

# train test split

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1,y1, test_size=1/3, random_state=72)


#%%

# =============================================================================
# Max alpha value and Min alpha value calculation for Dataset-1
# =============================================================================

alphaMAX1=0
alphaMIN1=0
r2MAX1=0.0001
r2MIN1=99

for i in range(0,1000):
    alp=i*.001
    R1 = Ridge(alpha=alp,fit_intercept=True, normalize=False, solver='saga',random_state=72)
    R1.fit(X_train1,y_train1)
    y_pred1  = R1.predict(X_test1)
    r_squared1 = metrics.r2_score(y_test1,y_pred1)
    
    if r_squared1>r2MAX1:
        alphaMAX1=alp
        r2MAX1=r_squared1
        
    if r_squared1<r2MIN1:
        alphaMIN1=alp
        r2MIN1=r_squared1

for i in range(1,1000):
    alp=i
    R1 = Ridge(alpha=alp,fit_intercept=True, normalize=False, solver='saga',random_state=72)
    R1.fit(X_train1,y_train1)
    y_pred1  = R1.predict(X_test1)
    r_squared1 = metrics.r2_score(y_test1,y_pred1)
    
    if r_squared1>r2MAX1:
        alphaMAX1=alp
        r2MAX1=r_squared1
        
    if r_squared1<r2MIN1:
        alphaMIN1=alp
        r2MIN1=r_squared1

print('\n___Dataset-1___')
print('\nAlpha Max :',alphaMAX1)
print('Alpha Min :',alphaMIN1)
print('R-Squred Max :',r2MAX1)
print('R-Squred Min :',r2MIN1)
print()

# ___Dataset-1___
# Alpha Max : 9
# Alpha Min : 999
# R-Squred Max : 0.9563174395769638
# R-Squred Min : 0.48662601447811027


#%%

R1 = Ridge(alpha=alphaMAX1, fit_intercept=True, normalize=False, solver='saga', random_state=72)
R1.fit(X_train1,y_train1)
y_pred1  = R1.predict(X_test1)


mae1 = metrics.mean_absolute_error(y_test1,y_pred1)
mse1 = metrics.mean_squared_error(y_test1,y_pred1)
rmse1 = metrics.mean_squared_error(y_test1,y_pred1)**.5
r_squared1 = metrics.r2_score(y_test1,y_pred1)

print('\nDataset-1:')
print('\nMean Absolute Error :',mae1)
print('Mean Squred Error :',mse1)
print('Root Mean Squred Error :',rmse1)
print('R-Squred :',r_squared1)
print()
print('score :',R1.score(X_train1, y_train1))
print('Intercept(a) :',R1.intercept_)
print('Coefficient(b) :\n',R1.coef_)
print()


#%%

# =============================================================================
#  Dataset-2
#  Independent variable put into X and Dependent variable 'OR' put into y
# =============================================================================

X2 = df2.iloc[:,:-1]
print(X2)

y2 = df2[['OR']]
print(y2)

#%%

# train test split

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2,y2, test_size=1/3, random_state=72)


#%%

# =============================================================================
# Max alpha value and Min alpha value calculation for Dataset-2
# =============================================================================

alphaMAX2=0
alphaMIN2=0
r2MAX2=0.0001
r2MIN2=99

for i in range(0,1000):
    alp=i*.001
    R2 = Ridge(alpha=alp,fit_intercept=True, normalize=False, solver='saga',random_state=42)
    R2.fit(X_train2,y_train2)
    y_pred2  = R2.predict(X_test2)
    r_squared2 = metrics.r2_score(y_test2,y_pred2)
    
    if r_squared2>r2MAX2:
        alphaMAX2=alp
        r2MAX2=r_squared2
        
    if r_squared2<r2MIN2:
        alphaMIN2=alp
        r2MIN2=r_squared2

for i in range(1,1000):
    alp=i
    R2 = Ridge(alpha=alp,fit_intercept=True, normalize=False, solver='saga',random_state=42)
    R2.fit(X_train2,y_train2)
    y_pred2  = R2.predict(X_test2)
    r_squared2 = metrics.r2_score(y_test2,y_pred2)
    
    if r_squared2>r2MAX2:
        alphaMAX2=alp
        r2MAX2=r_squared2
        
    if r_squared2<r2MIN2:
        alphaMIN2=alp
        r2MIN2=r_squared2

print('\n___Dataset-2___')
print('\nAlpha Max :',alphaMAX2)
print('Alpha Min :',alphaMIN2)
print('R-Squred Max :',r2MAX2)
print('R-Squred Min :',r2MIN2)
print()


# ___Dataset-2___
# Alpha Max : 48
# Alpha Min : 999
# R-Squred Max : 0.9624376197087541
# R-Squred Min : 0.9211670438661649


#%%

R2 = Ridge(alpha=alphaMAX2,fit_intercept=True, normalize=False, solver='saga',random_state=42)
R2.fit(X_train2,y_train2)
y_pred2  = R2.predict(X_test2)


mae2 = metrics.mean_absolute_error(y_test2,y_pred2)
mse2 = metrics.mean_squared_error(y_test2,y_pred2)
rmse2 = metrics.mean_squared_error(y_test2,y_pred2)**.5
r_squared2 = metrics.r2_score(y_test2,y_pred2)

print('\nDataset-2:')
print('\nMean Absolute Error :',mae2)
print('Mean Squred Error :',mse2)
print('Root Mean Squred Error :',rmse2)
print('R-Squred :',r_squared2)
print()
print('Score :',R2.score(X_train2, y_train2))
print('Intercept(a) :',R2.intercept_)
print('Coefficient(b) :\n',R2.coef_)
print()

#%%
