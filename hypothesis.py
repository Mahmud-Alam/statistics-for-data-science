xBar = float(input('Input XBar : '))
m = float(input('Input Mew (Mo) : '))
s = float(input('Input std : '))
n = float(input('Input (n) : '))

zScore = round((xBar-m)/(s/(n**.5)),5)

print('\nZ-score :',zScore)

#%%

z = float(input('Input Z-score : '))
m = float(input('Input Mew (Mo) : '))
s = float(input('Input std : '))
n = float(input('Input (n) : '))

x = round((z*(s/(n**.5)))+m,5)

print('\nX-bar :',x)