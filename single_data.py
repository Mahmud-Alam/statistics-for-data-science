n = int(input('Input number of n: '))
print('\nList of numbers: ')
x = [float(input()) for i in range(0,n)]

#%%

sumX = sum(x)
mean = round(sumX/n,4)
xMean = [x[i]-mean for i in range(0,n)]
xMeanSqr = [xMean[i]**2  for i in range(0,n)]
xMeanCube = [xMean[i]**3  for i in range(0,n)]
xMean4th = [xMean[i]**4  for i in range(0,n)]
sumXMeanSqr = sum(xMeanSqr)
sumXMeanCube = sum(xMeanCube)
sumXMean4th = sum(xMean4th)
variance = round(sumXMeanSqr/(n-1),4)
std = round(variance**.5,4)
skewness = round(sumXMeanCube/((n-1)*std**3),4)
kurtosis = round(sumXMean4th/((n-1)*std**4),4)

print('\nPrint Sorted array:')
x.sort()
print(x)

if(n%2==0):
    median = (x[int(n/2)-1]+x[int(n/2)])/2
else: median = x[int(n/2)]

Q1 = x[int(n*.25)-1]
Q2 = x[int(n*.5)-1]
Q3 = x[int(n*.75)-1]
IQR = Q3-Q1
Min = min(x)
Max = max(x)
lowerWhisker = Q1-(IQR*1.5)
upperWhisker = Q3+(IQR*1.5)

zScore = [round((x[i]-mean)/std,4) for i in range(0,n)]

print('\n Num ||x-mean||(x-mean)^2||(x-mean)^3||(x-mean)4||Z-score')
for i in range(0,n):
    print(x[i],'||',xMean[i],'||',xMeanSqr[i],'||',xMeanCube[i],'||',xMean4th[i],'||',zScore[i])

print('\n(x-mean)^2 :',sumXMeanSqr)
print('(x-mean)^3 :',sumXMeanCube)
print('(x-mean)^4 :',sumXMean4th)

print('\nMean :',mean)
print('Median :',median)
print('Variance :',variance)
print('Std :',std)
print('Skewness :',skewness)
print('Kurtosis :',kurtosis)

print('\nQ1 :',Q1)
print('Q2 :',Q2)
print('Q3 :',Q3)
print('IQR :',IQR)

print('\nMin :',Min)
print('Max :',Max)
print('Lower Whisker :',lowerWhisker)
print('Upper Whisker :',upperWhisker)

#%%

### T-Distribution

level = float(input('Input Confidence Level: '))

print('\nSum X:',sumX)
print('Mean :',mean)
print('(x-mean)^2 :',sumXMeanSqr)
print('Variance :',variance)
print('Std :',std)

alpha2 = (1-(level/100))/2
criticalLevel = 1-alpha2

print('\nArea Under the Curve : ',criticalLevel)
print('Degree of Freedom (DF) :',n-1)

tScore = float(input('Input t-Score: '))
lowV = round(mean-(tScore*(std/n**.5)),5)
highV = round(mean+(tScore*(std/n**.5)),5)

print('\n[',lowV,',',highV,']')

#%%

import numpy as np
import scipy.stats as st

data = np.array(x)
val = st.t.interval(alpha=level/100, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
print(val)


#%%

### If everything is given

xBar = float(input('Input XBar: '))
N = float(input('Input n: '))
s = float(input('Input std: '))
level = float(input('Input Confidence Level: '))

alpha2 = (1-(level/100))/2
criticalLevel = 1-alpha2

print('\nArea Under the Curve : ',criticalLevel)
print('Degree of Freedom (DF) :',n-1)

tS = float(input('Input t-Score: '))

lowVal = round(xBar-(tS*(s/N**.5)),5)
highVal = round(xBar+(tS*(s/N**.5)),5)

print('\n[',lowVal,',',highVal,']')


