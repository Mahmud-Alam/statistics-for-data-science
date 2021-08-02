n = int(input('Input number of N : '))
row = int(input('Input number of Row : '))
col = int(input('Input number of Col : '))
setA=[]; setB=[]; obs=[]; expt=[]

print('\nInput %percent set1(Row):')
[setA.append(round(float(input())/100,4)) for i in range(0,row)]
print('\nInput %percent set2(Column):')
[setB.append(round(float(input())/100,4)) for i in range(0,col)]

print(setA)
print(setB)

print('\nInput Observed values:')
for i in range(0,row):
    a=[]
    print('\n')
    print(i+1,'no row:')
    for j in range(0,col):
        print('\n',j+1,'no column:')
        a.append(float(input('Observed value: ')))
    obs.append(a)
    
#%%

for i in range(0,row):
    b=[]
    for j in range(0,col):
        b.append(setA[i]*setB[j]*n)
    expt.append(b)

print()
for i in range(0,row):
    for j in range(0,col):
        print('| ',round(obs[i][j],4),'(',round(expt[i][j],4),') |',end=(""))
    print()

#%%

chi_square=0

print()
for i in range(0,row):
    for j in range(0,col):
        chi_square+=(((obs[i][j]-expt[i][j])**2)/expt[i][j])

print('Chi-Square :',round(chi_square,5))

#%%

df = (row-1)*(col-1)
print('\nDF :',df)

confidence = float(input('Confidence Interval: '))
alfa = round((1-confidence/100)/2,4)
print('\nAlfa :',alfa)


