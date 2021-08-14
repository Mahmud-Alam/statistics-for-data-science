import pandas as pd
import numpy as np
import scipy.stats as st

#%%

data = np.array([45,55,67,45,68,79,98,87,84,82])

Z = st.zscore(data)

print(np.mean(data))
print(np.std(data))
val = st.t.interval(alpha=0.98, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))

print(val)

#%%

zTable = st.norm.interval(alpha=0.98, loc=np.mean(data), scale=st.sem(data))

print(st.sem(data))
print(zTable)

#%%

mean = round(data.mean(),6)
sd = round(data.std(),6)
print('\nXbar :',mean)
print('',sd)