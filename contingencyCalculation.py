row = int(input('Input number of Row : '))
col = int(input('Input number of Col : '))
obs = []; expt = []

print('\nInput Observed and Expected value:')
for i in range(0,row):
    a=[]; b=[]
    print()
    print(i+1,'no row:')
    for j in range(0,col):
        print('\n',j+1,'no column:')
        a.append(float(input('Observed value: ')))
        b.append(float(input('Expected value: ')))
    obs.append(a)
    expt.append(b)

print()
for i in range(0,row):
    for j in range(0,col):
        print('| ',obs[i][j],'(',expt[i][j],') |',end=(""))
    print()

#%%

chi_square=0

print()
for i in range(0,row):
    for j in range(0,col):
        chi_square+=(((obs[i][j]-expt[i][j])**2)/expt[i][j])

print('Chi-Square :',round(chi_square,4))