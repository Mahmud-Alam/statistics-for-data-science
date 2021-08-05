# =============================================================================
# Student ID: 2018-3-60-014
# Student Name: Md. Mahmud Alam
# Lab Assignment - 1
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('dataset_lab04.csv')
print(df.info())

#%%

# All functions

def Lab04_Task1_2018_3_60_014():
    rows = df.shape[0]
    columns = df.shape[1] 
    print('\nNumber of rows:',rows,'\nNumber of columns:',columns)


def Lab04_Task2_2018_3_60_014():
    print(round(df[['Time','Amount']].describe(),4))


def Lab04_Task3_2018_3_60_014():
    print("\nTime Column's Statistical Measures:")
    print('Mean:',round(df['Time'].mean(),5))
    print('Median:',round(df['Time'].median(),5))
    print('Standard Deviation:',round(df['Time'].std(),5))
    print('variance:',round(df['Time'].var(),5))
    
    print("\nAmount Column's Statistical Measures:")
    print('Mean:',round(df['Amount'].mean(),5))
    print('Median:',round(df['Amount'].median(),5))
    print('Standard Deviation:',round(df['Amount'].std(),5))
    print('variance:',round(df['Amount'].var(),5))


def Lab04_Task4_2018_3_60_014():
    time = df[['Time']]
    amount = df[['Amount']]
    df.boxplot(column=['Time','Amount'])
    
    print('\nTime Q3 :',time.quantile(.75))
    print('Time Median :',time.quantile(.5))
    print('Time Q1 :',time.quantile(.25))
    print('Time IQR :',time.quantile(.75)-time.quantile(.25))
    
    print('\nAmount Q3 :',amount.quantile(.75))
    print('Amount Median :',amount.quantile(.5))
    print('Amount Q1 :',amount.quantile(.25))
    print('Amount IQR :',amount.quantile(.75)-amount.quantile(.25))
    
    #For Outliers Checking
    
    print('\nTime Min :',time.min())
    print('Time Lower Bound :', time.quantile(.25)-(1.5*(time.quantile(.75)-time.quantile(.25))))
    print('Time Max :',time.max())
    print('Time Upper Bound :', time.quantile(.75)+(1.5*(time.quantile(.75)-time.quantile(.25))))
    
    print('\nAmount Min :',amount.min())
    print('Amount Lower Bound :', amount.quantile(.25)-(1.5*(amount.quantile(.75)-amount.quantile(.25))))
    print('Amount Max :',amount.max())
    print('Amount Upper Bound :', amount.quantile(.75)+(1.5*(amount.quantile(.75)-amount.quantile(.25))))
    

def Lab04_Task5_2018_3_60_014():
    time = df[['Time']]
    amount = df[['Amount']]

    # Time column Histogram
    time.hist(bins=50, color='lightgreen', edgecolor='black', linewidth=0.8)

    # Amount column Histogram
    amount.hist(bins=50, color='red', edgecolor='black', linewidth=0.8)
    
    # Skewness and Kurtosis
    print('\nTime Skewness :',time.skew())
    print('Time Kurtosis :',time.kurt())
    print('\nAmount Skewness :',amount.skew())
    print('Amount Kurtosis :',amount.kurt())
    
    
def Lab04_Task6_2018_3_60_014():
    print('\nPercentage of records with class value 0 and 1:')
    res = df[['Class']].value_counts(normalize=True)*100
    print(res)
    return res
    
    
def Lab04_Task7_2018_3_60_014():
    X = df['Class'].unique()
    Y = Lab04_Task6_2018_3_60_014()
    plt.bar(X, Y, width = 1, tick_label=["Non-Fraudulent","Fraudulent"],color=['lightgreen','darkred'],edgecolor=['black','darkred'],)
    plt.xlabel("Class Values")
    plt.ylabel("Value's Frequencies")
    plt.title("Class Column's Histogram")


def Lab04_Task8_2018_3_60_014():
    X = df['Class'].unique()
    Y = Lab04_Task6_2018_3_60_014()
    plt.bar(X, Y, width = 1, tick_label=["Non-Fraudulent","Fraudulent"],color=['lightgreen','darkred'],edgecolor=['black','darkred'],)
    plt.xlabel("Class Values")
    plt.ylabel("Value's Frequencies")
    plt.title("Class Column's Bar Chart")


def Lab04_Task9_2018_3_60_014():    
    print('\nV4 Skeweness:',round(df['V4'].skew(),5))
    print('V4 is positive skewed')
    df.hist(column=['V4'],bins=100, color='lightgreen', edgecolor='black', linewidth=0.5)
    
    print('\nV3 Skeweness:',round(df['V3'].skew(),5))
    print('V3 is negative skewed')
    df.hist(column=['V3'],bins=100, color='darkred', edgecolor='black', linewidth=0.5)

    print('V9 kurtosis:',round(df['V9'].kurt(),5))
    print('V9 is Leptokurtic kurtosis')
    df.hist(column=['V9'],bins=100, color='lightgreen', edgecolor='black', linewidth=0.5)

    print('Time kurtosis:',round(df['Time'].kurt(),5))
    print('Time is Platykurtic kurtosis')
    df.hist(column=['Time'],bins=100, color='darkred', edgecolor='black', linewidth=0.5)

    
def Lab04_Task10_2018_3_60_014():
    print('\nHighest Positive Correlation:')
    corr = df.corr()
    positive_corr = corr[corr<1]
    highest_positive_corr = positive_corr.max()
    print(highest_positive_corr)
    return highest_positive_corr
    
    
def Lab04_Task11_2018_3_60_014():
    highest_positive_corr = Lab04_Task10_2018_3_60_014()
    highest_positive_corr.plot.scatter(x='Time',y='Amount')
    
    
def Lab04_Task12_2018_3_60_014():
    print('\nHighest Negative Correlation:')
    corr = df.corr()
    highest_negative_corr = corr.min()
    print(highest_negative_corr)
    return highest_negative_corr
    

def Lab04_Task13_2018_3_60_014():
    highest_negative_corr = Lab04_Task12_2018_3_60_014()
    highest_negative_corr.plot.scatter(x='Time',y='Amount')
    

def Lab04_Task14_2018_3_60_014():
    amount = df[['Amount']]
    amount.boxplot()
    
    
def Lab04_Task15_2018_3_60_014():
    value0_df = df[['Amount','Class']].query('Class==0')
    print(value0_df)
    value0_series = value0_df['Amount']
    value1_df = df[['Amount','Class']].query('Class==1')
    print(value1_df)
    value1_series = value1_df['Amount']
    col = [value0_series, value1_series]
    figure, plot= plt.subplots()
    plot.boxplot(col)

        
    #%%
    
zero = df.loc[df['Class']==0]
one = df.loc[df['Class']==1]
print(zero.size/(df.size)*100)
print(one.size/(df.size)*100)
#%%
res = df['Class'].value_counts(normalize=True)*100
#res.plot(kind='bar')

X = df['Class'].unique()
Y = df[['Class']].value_counts(normalize=True)*100
plt.bar(X, Y, width = 0.6, tick_label=["Non-Fraudulent","Fraudulent"],color=['lightgreen','darkred'],edgecolor=['black','darkred'],)
plt.xlabel("Class Values")
plt.ylabel("Value's Frequencies")
plt.title("Class Column's Histogram")

    
    
#%%

x = df[['Time']]
plt.hist(x, bins=50, color=['lightgreen'],edgecolor='black',linewidth=0.6)
plt.xlabel('User Rating')
plt.ylabel('Number of Users')
plt.title('Customer Satisfaction Status')    

