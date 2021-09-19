import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# LENGTH CALCULATION FUNCTION
def LEN(LIST):
    List = np.array(LIST)
    n=0
    for i in List:
        n+=1    
    return n


# SUM CALCULATION FUNCTION
def SUM(LIST):
    List = np.array(LIST)
    if type(List[0])==str:
        st=''
        for i in List:
            st+=i
        return st
    SUM=0
    for i in List:
        SUM+=i    
    return SUM


# MEAN CALCULATION FUNCTION
def MEAN(List):
    return SUM(List)/LEN(List)


# ABSOLUTE CALCULATION FUNCTION
def ABSOLUTE(value):
    if (value<0):
        value*=-1
    return value


# Y-PREDICTION CALCULATION FUNCTION
def Y_PREDICTION(w, X):
    x = np.array(X)
    y_pred=[]
    for i in range(0,LEN(x[:,0])):
        val=0
        for j in range(0,LEN(w)):
            val+=w[j]*x[i][j]
        y_pred.append(val)
    return np.array(y_pred)


# MEAN ABSOLUTE ERROR CALCULATION FUNCTION
def MEAN_ABSOLUTE_ERROR(Y_PRED, Y_TEST):
    y_pred = np.array(Y_PRED)
    y_test = np.array(Y_TEST)
    return float(SUM([ABSOLUTE(y_pred[i]-y_test[i]) for i in range(0,LEN(y_test))])/LEN(y_test))


# MEAN SQUARED ERROR CALCULATION FUNCTION
def MEAN_SQUARED_ERROR(Y_PRED, Y_TEST):
    y_pred = np.array(Y_PRED)
    y_test = np.array(Y_TEST)
    return float(SUM([(y_pred[i]-y_test[i])**2 for i in range(0,LEN(y_test))])/LEN(y_test))


# R SQUARED CALCULATION FUNCTION
def R_SQUARED(Y_PRED, Y_TEST):
    y_pred = np.array(Y_PRED)
    y_test = np.array(Y_TEST)
    yMean = MEAN(y_test)
    return float(1-((SUM([(y_test[i]-y_pred[i])**2 for i in range(0,LEN(y_test))]))/(SUM([(y_test[i]-yMean)**2 for i in range(0,LEN(y_test))]))))
    

# MATRIX TRANSPOSE FUNCTION
def MATRIX_TRANSPOSE(matrix):
    transpose=[]
    for i in range(0,LEN(matrix[0,:])):
        temp=[]
        for j in range(0,LEN(matrix[:,0])):
            temp.append(matrix[j][i])
        transpose.append(temp)
    return np.array(transpose)


# MATRIX MULTIPLICATION FUNCTION
def MATRIX_MULTIPLICATION(matrix1, matrix2):
    new=[]
    for i in range(0, LEN(matrix1[:,0])):
        temp=[]
        for j in range(0, LEN(matrix2[0,:])):
            val=0
            for k in range(0, LEN(matrix1[0,:])):
                val += matrix1[i][k]*matrix2[k][j]
            temp.append(val)
        new.append(temp)
    return np.array(new)


# WEIGHT CALCULATION FUNCTION
def WEIGHT_FUNCTION(X, Y):
    x = np.array(X)
    y = np.array(Y)
    x_transpose = MATRIX_TRANSPOSE(x)
    x_mul = MATRIX_MULTIPLICATION(x_transpose, x)
    x_inv = np.linalg.inv(x_mul)
    y_mul = MATRIX_MULTIPLICATION(x_transpose, y)
    w = MATRIX_MULTIPLICATION(x_inv, y_mul)
    return w


#%%

# Read CSV File

df_actual = pd.read_csv('student_satisfaction.csv')
print(df_actual.head())
print(df_actual.shape)

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

# Add B0 column with all value 1

df.insert(0,'B0',1)
print(df.head())

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

# Independent variable put into X and Dependent variable 'OR' put into y

X = df.iloc[:,:-1]
print(X)

y = df[['OR']]
print(y)

#%%

# train test split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=1/3, random_state=72)

#%%


weight = WEIGHT_FUNCTION(X_train, y_train)

print(weight)
#%%

y_pred = Y_PREDICTION(weight, X_test)
print(y_pred)

#%%

mae = MEAN_ABSOLUTE_ERROR(y_pred, y_test)
mse = MEAN_SQUARED_ERROR(y_pred, y_test)
rmse = MEAN_SQUARED_ERROR(y_pred, y_test)**.5
r_squared = R_SQUARED(y_pred, y_test)


print('\nCalculation Results:')
print('\nMean Absolute Error :',mae)
print('Mean Squred Error :',mse)
print('Root Mean Squred Error :',rmse)
print('R-Squred :',r_squared)
print()
#print('Score :',score(X_train, y_train))
