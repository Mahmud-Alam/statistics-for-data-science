n = int(input('Input number of n: '))
print('\nList of X: ')
x = [float(input()) for i in range(0,n)]
print('\nList of Y: ')
y = [float(input()) for i in range(0,n)]

#%%

sumX = sum(x)
sumY = sum(y)
meanX = round(sumX/n,4)
meanY = round(sumY/n,4)

xMeanX = [x[i]-meanX for i in range(0,n)]
yMeanY = [y[i]-meanY for i in range(0,n)]
xMeanXSqr = [xMeanX[i]**2 for i in range(0,n)]
yMeanYSqr = [yMeanY[i]**2 for i in range(0,n)]
sumxMeanXSqr = sum(xMeanXSqr)
sumyMeanYSqr = sum(yMeanYSqr)

varX = round(sumxMeanXSqr/(n-1),4)
varY = round(sumyMeanYSqr/(n-1),4)
stdX = round(varX**.5,4)
stdY = round(varY**.5,4)

xMeanXymeanY = [xMeanX[i]*yMeanY[i] for i in range(0,n)]
sumxMeanXymeanY = sum(xMeanXymeanY)

# Pearson's Correlation Coefficiant
corr = round(sumxMeanXymeanY/((n-1)*stdX*stdY),4)

# Covairance
covairance = round(sumxMeanXymeanY/n,4)
rXY = round(covairance/(stdX*stdY),4)

print('\n  X  ||   Y   || x-Xbar || y-Ybar ||(x-Xbar)^2||(y-Ybar)^2||(x-Xbar)(y-Ybar)')
for i in range(0,n):
    print(x[i],'||',y[i],'||',round(xMeanX[i],4),'||',round(yMeanY[i],4),'||',round(xMeanXSqr[i],4),'||',round(yMeanYSqr[i],4),'||',round(xMeanXymeanY[i],4))

print('\nMean X :',meanX)
print('Mean Y :',meanY)
print('Sum of (x-Xbar)^2 :',sumxMeanXSqr)
print('Sum of (y-Ybar)^2 :',sumyMeanYSqr)
print('Sum of (x-Xbar)(y-Ybar) :',sumxMeanXymeanY)

print('\nVariance X :',varX)
print('Std X :',stdX)
print('Variance Y :',varY)
print('Std Y :',stdY)
print('Correlation Coefficient (r) :',corr)

print('\nCovariance :',covairance)
print('rXY :',rXY)

if(corr==1):
    print('\nr = ',corr,'\nSo, They are in Perfect Positive Correlation.')
elif(corr<1 and corr>=0.9):
    print('\nr = ',corr,'\nSo, They are in High Positive Correlation.')
elif(corr<0.9 and corr>=0.5):
    print('\nr = ',corr,'\nSo, They are in Low Positive Correlation.')
elif(corr<0.5 and corr>0):
    print('\nr = ',corr,'\nSo, They are in Very Low Positive Correlation.')
elif(corr==0):
    print('\nr = ',corr,'\nSo, They have NO Correlation.')
elif(corr<0 and corr>-0.5):
    print('\nr = ',corr,'\nSo, They are in Very Low Negative Correlation.')
elif(corr<=-0.5 and corr>-0.9):
    print('\nr = ',corr,'\nSo, They are in Low Negative Correlation.')
elif(corr<=-0.9 and corr>-1):
    print('\nr = ',corr,'\nSo, They are in High Negative Correlation.')
elif(corr==-1):
    print('\nr = ',corr,'\nSo, They are in Perfect Negative Correlation.')
else: print('\nCorrelation value is out of range!')

