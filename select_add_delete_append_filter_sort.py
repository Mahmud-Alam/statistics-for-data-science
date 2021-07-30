import pandas as pd

#%%
list1 = [['Alice',23,3.5,10],['Bob',24,3.4,6],['Charlie',22,3.9,8]]
df = pd.DataFrame(list1)
df.columns = ['name','age','cgpa','hoursStudied']

#%%

print(df)

#%%

print(df.iloc[0][0])
print(df.iloc[0][3])

#%%

#create a new dataframe and containing first two rows with all columns

#1st way
df1 = df.iloc[0:2,:]
print(df1)

#2nd way
df2 = df.iloc[:2,:]
print(df2)

#3rd way
df3 = df.iloc[:2]
print(df3)

#%%

#create a new dataframe and containing last two rows with all columns

#1st way
df4 = df.iloc[-2:,:]
print(df4)

#2nd way
df5 = df.iloc[-2:]
print(df5)

#%%

#create a new dataframe and containing last two rows with last two columns
df6 = df.iloc[-2:,-2:]
print(df6)

#%%

# Include last n columns
df6 = df.iloc[:,-2:]
print(df6)
print()

# Exclude last n columns
df62 = df.iloc[:,:-2]
print(df62)


#%%

#particular rows all columns
#1st way
print(df.iloc[1])
#2nd way
print(df.iloc[1,:])
#3rd way
print(df.iloc[1:2])
#4th way
print(df.iloc[1:2,:])

#%%

#particular columns all rows
print(df.iloc[:,0])

#%%

#create a dataframe with first two columns with all rows 
df7 = df.iloc[:,:2]
print(df7)

#%%

#create a dataframe with last two columns with all rows 
df8 = df.iloc[:,-2:]
print(df8)

#%%

#create a dataframe with any distanced columns with last two rows 
df8 = df.iloc[-2:,[0,2,3]]
print(df8)

#%%

# loc method

# find one element location with loc method
df9 = df.loc[0]['name']
print(df9)

#%%
#particular columns all rows with loc method
df9 = df.loc[:,'name']
print(df9)
#%%

#create a dataframe with first two columns with all rows with loc method
df9 = df.loc[:,('name','age')]
print(df9)

print()
df10 = df.loc[:,['name','age']]
print(df10)

#%%
#same work with iloc method
df5 = df.iloc[1:3,[0,1]]
print(df5)

#%%

#create a dataframe with first two columns with first two rows
# 1 dile, loc 0,1 both count kore bt iloc e 1 dile, only 0 count korbe.
 
df11 = df.loc[:1,('name','age')]
print(df11)

#%%

#create a dataframe with first two columns with last two rows
# 1 dile, loc 0,1 both count kore bt iloc e 1 dile, only 0 count korbe.

df12 = df.loc[-2:,('name','age')]
print(df12)

#NOT WORKING
#NOT WORKING
#NOT WORKING
#NOT WORKING

#%%

#create any two columns with all rows
df13 = df.loc[:,('name','hoursStudied')]
print(df13)

#%%

#ADDing a column into dataframe

list2 = [10,20,30]
df['id'] = pd.DataFrame(list2)
print(df)

#%%

#Drop a column from dataframe 

df.drop(['id'],axis=1,inplace=True)
print(df)

#%%

df14 = df.copy()
print(df14)

#%%

list3 = ['female','male','male']
df14['gender'] = pd.DataFrame(list3)
print(df14)
print(type(df14))

#%%

#row delete

df15 = df.copy()
df15.drop([0,1], inplace=True)
print(df15)

#%%

# Append two dataframe together

df16 = df.copy()
df17 = df.copy()

print(df16)
print()
print(df17)
print()
df18 = df16.append(df17)
print(df18)

#%%

# Append with different rows (2x2)+(3x2)

df19 = df.iloc[:2,:2]
print(df19)
df20 = df.iloc[:,:2]
print(df20)
print()
df21 = df19.append(df20)
print(df21)

#%%

# Append with different columns (2x2)+(3x4)

df22 = df.iloc[:2,:2]
print(df22)
df23 = df.iloc[-3:,:]
print(df23)
print()
df25 = df22.append(df23, ignore_index=True)
print(df25)

#%%

# Data Filtering with series

all_females = df14['gender']=='female'
print(all_females)
print(type(all_females))

#%%

# Data Filtering from DataFrame

all_females = df14[df14['gender']=='female']
print(all_females)
print(type(all_females))

#%%

# Data Filtering with loc function

all_females1 = df14.loc[df14['gender']=='female']
print(all_females1)

#%%

# Data Filtering with query

all_females2 = df14.query('gender =="female"')
print(all_females2)

#%%

# different condition with query
# retrive all the recordes with gender male and cgpa more than 3.5

top_students = df14.query('gender =="male" and cgpa>3.5')
print(top_students)

#%%

# different condition from dataframe
# retrive all the recordes with gender male and cgpa more than 3.5

top_students1 = df14[(df14['gender']=='male') & (df14['cgpa']>3.5)]
print(top_students1)
print(type(top_students1))

#%%

# different condition with loc function
# retrive all the recordes with gender male and cgpa more than 3.5

top_students2 = df14.loc[(df14['gender']=='male') & (df14['cgpa']>3.5)]
print(top_students2)
print(type(top_students2))

#%%

# Sorting and inplace all sort values

print(df)                                               #Unsorted
df.sort_values(by='age',ascending=False, inplace=True)  # sorted and saved inplace
print()
print(df)                                               #Sorted because inplace saved

#%%

# Sorting and without inplace sort values

print(df25)                                            # Unsorted values
print()
print(df.sort_values(by='age',ascending=False))        # Sorted but not saved inplace
print()
print(df25)                                            # Still unsorted because not saved inplace 