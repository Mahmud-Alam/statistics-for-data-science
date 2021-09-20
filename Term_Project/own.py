import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# LENGTH CALCULATION FUNCTION
def LEN(List):
    n=0
    for i in List:
        n+=1    
    return n


# SUM CALCULATION FUNCTION
def SUM(List):
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
def Y_PREDICTION(w, x):
    y_pred=[]
    for i in range(0,LEN(x[:,0])):
        val=0
        for j in range(0,LEN(w)):
            val+=w[j]*x[i][j]
        y_pred.append(val)
    return np.array(y_pred)


# MEAN ABSOLUTE ERROR CALCULATION FUNCTION
def MEAN_ABSOLUTE_ERROR(y_pred, y_test):
    return SUM([ABSOLUTE(y_pred[i]-y_test[i]) for i in range(0,LEN(y_test))])/LEN(y_test)


# MEAN SQUARED ERROR CALCULATION FUNCTION
def MEAN_SQUARED_ERROR(y_pred, y_test):
    return SUM([(y_pred[i]-y_test[i])**2 for i in range(0,LEN(y_test))])/LEN(y_test)


# R SQUARED CALCULATION FUNCTION
def R_SQUARED(y_pred, y_test, yMean):
    return 1-((SUM([(y_test[i]-y_pred[i])**2 for i in range(0,LEN(y_test))]))/(SUM([(y_test[i]-yMean)**2 for i in range(0,LEN(y_test))])))
    

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
def WEIGHT_FUNCTION(x, y):
    x_transpose = MATRIX_TRANSPOSE(x)
    x_mul = MATRIX_MULTIPLICATION(x_transpose, x)
    x_inv = np.linalg.inv(x_mul)
    y_mul = MATRIX_MULTIPLICATION(x_transpose, y)
    w = MATRIX_MULTIPLICATION(x_inv, y_mul)
    return w


#%%

df = pd.read_csv('student_satisfaction.csv')
print(df.head())
print(df.shape)

#%%

df.drop(['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','Satisfied'],axis=1, inplace=True)

print(df)

#%%

print(df.isnull().sum())

#%%

df['IA1'].fillna(df['IA1'].median(skipna=True), inplace=True)
df['IA3'].fillna(df['IA3'].median(skipna=True), inplace=True)
df['TLA2'].fillna(df['TLA2'].median(skipna=True), inplace=True)

#%%

X = df.drop(['OR'],axis=1)
print(X)

y = df[['OR']]
print(y)

#%%

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=1/3, random_state=1)

X_TRAIN = np.array(X_train)
X_TEST = np.array(X_test)
Y_TRAIN = np.array(y_train)
Y_TEST = np.array(y_test)

#%%

weight = WEIGHT_FUNCTION(X_TRAIN, Y_TRAIN)

print(weight)

#%%

Y_PRED = Y_PREDICTION(weight, X_TEST)
print(Y_PRED)

#%%

mae = MEAN_ABSOLUTE_ERROR(Y_PRED, Y_TEST)
mse = MEAN_SQUARED_ERROR(Y_PRED, Y_TEST)
rmse = MEAN_SQUARED_ERROR(Y_PRED, Y_TEST)**.5
r_squared = R_SQUARED(Y_PRED, Y_TEST,MEAN(Y_TEST))


print('\nCalculation Results:')
print('\nMean Absolute Error :',mae)
print('Mean Squred Error :',mse)
print('Root Mean Squred Error :',rmse)
print('R-Squred :',r_squared)
print()