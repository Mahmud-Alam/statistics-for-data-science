import pandas as pd

#%%

df = pd.read_csv("updated_student_info.csv")
print(df)

#%%

age = df[['Age']]
print(age)

#%%

print(age.sum())

#%%

def sum_Group1(df):
    s=0
    print(type(df))
    
    if isinstance(df, pd.DataFrame):
        df = df.iloc[:,0]
    
    for i in df:
        s+=i
        
    print(type(df))    
    return s



#%%

age = df[['Age']]
print(age)

print(sum_Group1(age))


#%%

a = df['Age']
print(a)

print(sum_Group1(a))

