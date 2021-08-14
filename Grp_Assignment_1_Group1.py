import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#%%

df = pd.read_csv('german-data.csv',header=None)
df.columns = ['Status of existing checking account','Duration in month','Credit history','Purpose','Credit amount','Savings account/bonds','Present employment since','Installment rate in percentage of disposable income','Personal status and sex','Other debtors / guarantors','Present residence since','Property','Age in years','Other installment plans','Housing','Number of existing credits at this bank','Job','Number of people being liable to provide maintenance for','Telephone','foreign worker','Cost Matrix']

print(df.info())

#%%

df_num = pd.read_csv('german-data-numeric.csv',header=None)
df_num.columns = ['Column-01','Column-02','Column-03','Column-04','Column-05','Column-06','Column-07','Column-08','Column-09','Column-10','Column-11','Column-12','Column-13','Column-14','Column-15','Column-16','Column-17','Column-18','Column-19','Column-20','Column-21','Column-22','Column-23','Column-24','Column-25',]

print(df_num.info())

#%%

df.to_csv('updated-german-data.csv',index=False)
df_num.to_csv('updated-german-data-numeric.csv',index=False)

#%%

job = df[['Job']]
jobCount = job.value_counts()
jobCount = jobCount.sort_index()
jobTitle = sorted(df['Job'].unique())

print(jobCount)
print(jobTitle)
labels = ['Unemployed','Unskilled','Skilled','Highly Qualified']
colors = ['pink','red','crimson','darkred']

plt.figure(figsize=(8,6),dpi=500)

plt.subplot(2,2, 1)
plt.xticks(fontsize=7)
plt.bar(jobTitle, jobCount,tick_label = labels, edgecolor='black', color=colors)
plt.title('Job Bar-Chart')

plt.subplot(2,2, 2)
percent = df['Job'].value_counts().sort_index() / df['Job'].value_counts().sum() * 100
plt.pie(percent, labels=labels, explode=(0, 0, 0, 0.15), autopct='%1.1f%%',startangle=10, colors=['lightgreen','orange','yellow','crimson'])
plt.axis('equal')
plt.title('Job Pie-Chart')


plt.subplot(2,2, 3)
plt.xticks(fontsize=7)
plt.bar(jobTitle, percent, tick_label = labels, width=1,edgecolor='black', color='lightgreen')
plt.title('Job Histogram')


plt.subplot(2,2, 4)
plt.plot(labels,jobCount, 'rv--',markersize=10)
plt.title('Job Plot Diagram')

#%%

amount = df[['Credit amount']]

plt.figure(dpi=500)
gs = gridspec.GridSpec(7, 7)

plt.subplot(gs[:3, :3])
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.hist(amount, bins=50,edgecolor='black', color='red')
plt.text(7000, 70, 'median<mean', color='darkred',fontsize=8)
plt.xlabel('Amount',fontsize=8)
plt.ylabel('Frequency',fontsize=8)
plt.title('Credit-Amount Histogram',fontsize=10)

plt.subplot(gs[:, 4:])
plt.boxplot(amount)
plt.title('Credit-Amount Boxplot',fontsize=10)

plt.subplot(gs[4:,:3])
plt.scatter(amount.index,amount, color='red',marker='2')

