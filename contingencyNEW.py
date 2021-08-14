import numpy as np

#%%

row = int(input('Input number of Row : '))
col = int(input('Input number of Col : '))

ob=[]

print('\nInput Observed values:')
for i in range(0,row):
    a=[]
    print('\n')
    print(i+1,'no row:')
    for j in range(0,col):
        a.append(float(input('Observed value: ')))
    ob.append(a)
    
#%%

obs = np.array(ob)
print(obs)

n = np.sum(obs)
sumColumn = np.sum(obs, axis=0)
sumRow = np.sum(obs, axis=1)

print('\ntotal data (n) :',n)
print('\nSum Column :',sumColumn)
print('\nSum Row :',sumRow)
ex=[]
for i in range(0,row):
    a=[]
    for j in range(0,col):
        a.append(round((sumColumn[j]*sumRow[i])/n,4))
    ex.append(a)

exp = np.array(ex)
print('\nContingency Table : ')
for i in range(0,row):
    for j in range(0,col):
        print('|',obs[i][j],' ( ',exp[i][j],' ) |',end='')
    print()

#%%

chi_square=0

print()
for i in range(0,row):
    for j in range(0,col):
        chi = (((obs[i][j]-exp[i][j])**2)/exp[i][j])
        chi_square+=chi
        print(chi)

print('\nChi-Square :',round(chi_square,6))

#%%

df = (row-1)*(col-1)
print('\nDF :',df)

confidence = float(input('Confidence Interval: '))
alfa = round((1-confidence/100)/2,4)
print('\nAlfa :',alfa)


