l = lower = int(input('Input Lower Bound: '))
upper = int(input('Input Upper Bound: '))
w = int(input('Input Interval: '))

a = int((upper-lower)/w)

print('\nList of Frequency: ')
frq = [int(input()) for i in range(0,a)]

#%%
l = lower
l = l-w
setA = [l:=l+w for i in range(0,a)]
#print(setA)

l = lower
setB = [l:=l+w for i in range(0,a)]
#print(setB)

mid = [(setA[i]+setB[i])/2 for i in range(0,a)]
#print(mid)

frqMid = [frq[i]*mid[i] for i in range(0,a)]
#print(frqMid)

cumu = [0]*a
cumu[0] = frq[0]
k=0
for i in range(1,a):
    cumu[i]=cumu[k]+frq[i]
    k+=1
#print(cumu)

n = sum(frq)
sumFrqMid = sum(frqMid)
mean = round(sumFrqMid/n,4)
#print(mean)

midXbar = [mid[i]-mean for i in range(0,a)]
frqMidXbarSrq = [frq[i]*midXbar[i]**2 for i in range(0,a)]
sumFrqMidXbarSqr = round(sum(frqMidXbarSrq),4)
#print(sumFrqMidXbarSqr)

n2 = n/2
for i in range(0,a):
    if cumu[i]>=n2:
        B = cumu[i-1]
        L = setA[i]
        G = frq[i]
        break

        
median = round(L+((n2-B)/G)*w,4)
variance = round(sumFrqMidXbarSqr/(n-1),4)
std = round(variance**.5,4)


frqMidXbarCube = [frq[i]*midXbar[i]**3 for i in range(0,a)]
frqMidXbar4th = [frq[i]*midXbar[i]**4 for i in range(0,a)]
sumFrqMidXbarCube = round(sum(frqMidXbarCube),4)
sumFrqMidXbar4th = round(sum(frqMidXbar4th),4)
skewness = round(sumFrqMidXbarCube/((n-1)*std**3),4)
kurtosis = round(sumFrqMidXbar4th/((n-1)*std**4),4)

print('\nInterval|| Frq || Mid || frq*mid || Cumu ||mid - Xbar||f(x-Xbar)^2||f(x-Xbar)^3||f(x-Xbar)^4')
for i in range(0,a):
    print(setA[i],'-',setB[i],'|| ',frq[i],' || ',mid[i],' || ',round(frqMid[i],4),' || ',round(cumu[i],4),' || ',round(midXbar[i],4),' || ',round(frqMidXbarSrq[i],4),' || ',round(frqMidXbarCube[i],4),' || ',round(frqMidXbar4th[i],4))

print('\nSum Frq :',n)
print('Sum Frq*Mid :',sumFrqMid)
print('Sum f(x-Xbar)^2 :',sumFrqMidXbarSqr)
print('Sum f(x-Xbar)^3 :',sumFrqMidXbarCube)
print('Sum f(x-Xbar)^4 :',sumFrqMidXbar4th)

print('\nL :',L)
print('n/2 :',n2)
print('B :',B)
print('G :',G)
print('W :',w)

print('\nMean :',mean)
print('Median :',median)
print('Variance :',variance)
print('Standard Deviation :',std)
print('Skewness :',skewness)
print('Kurtosis :',kurtosis)