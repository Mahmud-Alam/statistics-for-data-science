import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#%%

### Categorical Data

df_cat = pd.read_csv('german-data.csv',header=None)
df_cat.columns = ['Status of existing checking account','Duration in month','Credit history','Purpose','Credit amount','Savings account/bonds','Present employment since','Installment rate in percentage of disposable income','Personal status and sex','Other debtors / guarantors','Present residence since','Property','Age in years','Other installment plans','Housing','Number of existing credits at this bank','Job','Number of people being liable to provide maintenance for','Telephone','foreign worker','Cost Matrix']

print(df_cat.info())

#%%

### Numeric Data

df_num = pd.read_csv('german-data-numeric.csv',header=None)
df_num.columns = ['Status of existing checking account','Duration in month','Credit history','Purpose','Credit amount','Savings account/bonds','Present employment since','Installment rate in percentage of disposable income','Personal status and sex','Other debtors / guarantors','Present residence since','Property','Age in years','Other installment plans','Housing','Number of existing credits at this bank','Job','Number of people being liable to provide maintenance for','Telephone','foreign worker','Cost Matrix','Column-22','Column-23','Column-24','Column-25',]

print(df_num.info())

#%%

df_num.to_csv('updated-german-data.csv',index=False)
df_num.to_csv('updated-german-data-numeric.csv',index=False)

#%%
#Khaled
#1.1 (Plot-01)


ax=df_num['Personal status and sex'].hist(color="g")
plt.xlabel('Staus and sex',fontsize=14,color="Blue" )
plt.ylabel('Frequency of sex gender',fontsize=14,color="Red")
plt.xticks([1,2,3,4,5],['Male divorced','Female Divorced','Male Single','Male Married','Female single'])

plt.title("Personal Status and Sex",fontsize=15,color="Cyan")
plt.grid()
plt.show()


#%%
#1.2 (Plot-02)

plt.scatter(df_num['Present employment since'],df_num["Purpose"],color="r",marker='x')
plt.xlabel("Present employment",fontsize=14,color="orange")
plt.ylabel("Purpose",fontsize=14,color="magenta") 
plt.title("Scatter diagram of Employeee and Purpose",fontsize=14,color="green")

plt.figure()

#%%
#used it to see the corelations
import seaborn as sns
plt.subplots(figsize=(12, 9))
sns.heatmap(df_num.corr(), annot = True, cmap = 'cividis')


#%%

# =============================================================================
# Name: Md. Mahmud Alam
# Id: 2018-3-60-014
# 2.1 (Plot-01)
# =============================================================================

job = df_cat[['Job']]
jobCount = job.value_counts()
jobCount = jobCount.sort_index()
jobTitle = sorted(df_cat['Job'].unique())

labels = ['Unemployed','Unskilled','Skilled','Highly Qualified']
colors = ['pink','red','crimson','darkred']

plt.figure(figsize=(8,6),dpi=500)

plt.subplot(2,2,1)
percent = df_cat['Job'].value_counts().sort_index() / df_cat['Job'].value_counts().sum() * 100
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.text(-.2, 7, '2.2%', color='darkgreen',fontsize=8)
plt.text(.82, 25, '20%', color='darkgreen',fontsize=8)
plt.text(1.82, 55, '63%', color='darkgreen',fontsize=8)
plt.text(2.75, 20, '14.8%', color='darkgreen',fontsize=8)
plt.bar(jobTitle, percent, tick_label = labels, width=1,edgecolor='darkgreen', color='lightgreen')
plt.title('Job Histogram',fontsize=10)


plt.subplot(2,2,2)
plt.pie(percent, labels=labels, explode=(0, 0, 0, 0.15), autopct='%1.1f%%',startangle=10, colors=['lightgreen','orange','yellow','crimson'])
plt.axis('equal')
plt.title('Job Pie-Chart',fontsize=10)


plt.subplot(2,2,3)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.text(-.1, 70, '22', color='black',fontsize=8)
plt.text(.85, 250, '200', color='black',fontsize=8)
plt.text(1.85, 550, '630', color='black',fontsize=8)
plt.text(2.85, 200, '148', color='black',fontsize=8)
plt.bar(jobTitle, jobCount,tick_label = labels, edgecolor='black', color=colors)
plt.title('Job Bar-Chart',fontsize=10)

plt.subplot(2,2,4)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.plot(labels,jobCount, 'rv--',markersize=10)
plt.title('Job Plot Diagram',fontsize=10)


#%%

# =============================================================================
# Name: Md. Mahmud Alam
# Id: 2018-3-60-014
# 2.2 (Plot-02)
# =============================================================================

amount = df_cat[['Credit amount']]

plt.figure(dpi=500)
gs = gridspec.GridSpec(7, 7)

plt.subplot(gs[:3, :3])
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.hist(amount, bins=50,edgecolor='black', color='red')
plt.text(5000, 100, 'Positive Skewed Diagram', color='darkred',fontsize=7)
plt.text(8000, 85, 'median<mean', color='red',fontsize=7)
plt.xlabel('Amount',fontsize=8)
plt.ylabel('Frequency',fontsize=8)
plt.title('Credit-Amount Histogram',fontsize=10)

plt.subplot(gs[:, 4:])
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.boxplot(amount)
plt.title('Credit-Amount Boxplot',fontsize=10)
plt.text(1.1, 11000, 'Outliers', color='red',fontsize=8)
plt.text(1.12, 7600, 'Max', color='red',fontsize=8)
plt.text(1.12, 3700, 'Q3', color='red',fontsize=8)
plt.text(1.12, 2100, 'Median', color='red',fontsize=8)
plt.text(1.12, 1100, 'Q1', color='red',fontsize=8)
plt.text(1.12, 1, 'Min', color='red',fontsize=8)

plt.subplot(gs[4:,:3])
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.scatter(amount.index,amount, color='red',marker='2')
plt.title('Credit-Amount Scatter Plot',fontsize=10)


#%%
#Talha
#3.1 (Plot-01)
#Installment rate in percentage of disposable income

installment = df_num['Present employment since']
#credit = df_num['Column-04']
#plt.boxplot(installment)
df_num.boxplot(column = ['Present employment since'])

#%%
#3.2 (Plot-02)
labels = ['Bank','store','none']
other = df_cat['Other installment plans'].value_counts()
#plot(kind='pie')

fig, f = plt.subplots()
f.pie(other, labels = labels)


