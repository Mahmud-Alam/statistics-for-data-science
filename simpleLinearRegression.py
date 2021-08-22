n = int(input('How many numbers: '))

setX=[]
setY=[]

print('\nInput set X values:')
for i in range(0,n):
    setX.append(float(input()))

print('\nInput set Y values:')
for i in range(0,n):
    setY.append(float(input()))

#%%

xBar = sum(setX)/n
yBar = sum(setY)/n

xMinusXbar = [setX[i]-xBar for i in range(0,n)]
yMinusYbar = [setY[i]-yBar for i in range(0,n)]
xMXbarYMYbar = [xMinusXbar[i]*yMinusYbar[i] for i in range(0,n)]
xMinusXbarSqr = [xMinusXbar[i]**2 for i in range(0,n)]

cov = sum(xMXbarYMYbar)/(n-1)
var = sum(xMinusXbarSqr)/(n-1)

b = cov/var
a = yBar-(cov/var)*xBar

print('\nXBar :',xBar)
print('YBar :',yBar)
print('Sum of (X-xBar)^2 :',sum(xMinusXbarSqr))
print('Sum of (X-xBar)(Yhat-yBar) :',sum(xMXbarYMYbar))

print('\nCov(X,Yhat) :',cov)
print('Var(X) :',var)

print('\na (Y intersect):',round(a,5))
print('b (slope)      :',round(b,5))

print('\nEquation:')
print('Y = a + bX')
print('Y = ',round(a,5),' + ',round(b,5),'X')

print('\nTable:')
print('_____________________________________________________________________________________')
print('|    X    ||   Yhat  ||   X-xBar   ||  Yhat-yBar || (X-xBar)^2 ||(X-xBar)(Yhat-yBar)|')
print('|_________||_________||____________||____________||____________||___________________|')
for i in  range(0,n):
    print('| {:7} || {:7} || {:10} || {:10} || {:10} || {:17} |'.format(setX[i],setY[i],round(xMinusXbar[i],4),round(yMinusYbar[i],4),round(xMinusXbarSqr[i],4),round(xMXbarYMYbar[i],4)))
print('|_________||_________||____________||____________||____________||___________________|')

#%%

predY = [a+(b*setX[i]) for i in range(0,n)]
residual = [abs(predY[i]-setY[i]) for i in range(0,n)]

print('\nTable:')
print('__________________________________________________')
print('|    X    ||   Yhat  ||   pred Y   ||   residual |')
print('|_________||_________||____________||____________|')
for i in  range(0,n):
    print('| {:7} || {:7} || {:10} || {:10} |'.format(setX[i],setY[i],round(predY[i],4),round(residual[i],4)))
print('|_________||_________||____________||____________|')

