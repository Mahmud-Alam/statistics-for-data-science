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
    print('\n\033[1;33;1m Number of rows: \033[0m',rows,'\n\033[1;33;1m Number of cols: \033[0m',columns)


def Lab04_Task2_2018_3_60_014():
    print(round(df[['Time','Amount']].describe(),4))


def Lab04_Task3_2018_3_60_014():
    print("\n\033[1;33;1mTime Column's Statistical Measures:\033[0m")
    print('Mean:',round(df['Time'].mean(),5))
    print('Median:',round(df['Time'].median(),5))
    print('Standard Deviation:',round(df['Time'].std(),5))
    print('variance:',round(df['Time'].var(),5))
    
    print("\n\033[1;33;1mAmount Column's Statistical Measures:\033[0m")
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
    
    print('\n\033[1;33;1mIn Time column, there is no outliers\nbut Amount column has outliers.\033[0m')
    
    # Answer Explanation
    
    print('\n\033[1;33;1mAnswer Explanation:\n\033[0m')
    print('For Time column \nMin>Lower Bound and max<Upper Bound. So, Time Columns has no Outliers. Also we can see from the box plot, that median<mean so Time column is Positively skewed.')
    print('\nFor Amount column \nMin>Lower Bound but max>Upper Bound. So, Amount Columns has Outliers. Also we can see from the box plot, that median<mean so Amount column is Positively skewed.')


def Lab04_Task5_2018_3_60_014():
    time = df[['Time']]
    amount = df[['Amount']]

    # Time column Histogram
    time.hist(bins=50, color='lightgreen', edgecolor='black', linewidth=0.8)

    # Amount column Histogram
    amount.hist(bins=50, color='red', edgecolor='black', linewidth=0.8)
    
    # Skewness and Kurtosis
    print("\n\033[1;33;1mTime column's Skewness and Kurtosis values:\033[0m")
    print('Time Skewness :',time.skew())
    print('Time Kurtosis :',time.kurt())
    print("\n\033[1;33;1mAmount column's Skewness and Kurtosis values:\033[0m")
    print('Amount Skewness :',amount.skew())
    print('Amount Kurtosis :',amount.kurt())

    # Comment on Data Distribution:
    print('\n\033[1;33;1mComment on Data Distribution:\n\033[0m')
    print('For Time column \nSkewness value is negative so Time is negative skewed distribution. also Kurtosis value is negative, so it is Platykurtic kurtosis, it means there are many small numbers of extreme values and possiblities of more outliers.')
    print('\nFor Amount column \nSkewness value is positive so Amount is positive skewed distribution. also Kurtosis value is highly positive, so it is Leptokurtic kurtosis, it means there are less numbers of extreme values, but in the middle there are very higher number of median and mean values.')

    
def Lab04_Task6_2018_3_60_014():
    print('\n\033[1;33;1mPercentage of records with \nclass value 0 and 1:\033[0m')
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
    plt.bar(X, Y, width = 0.6, tick_label=["Non-Fraudulent","Fraudulent"],color=['lightgreen','darkred'],edgecolor=['black','darkred'],)
    plt.xlabel("Class Values")
    plt.ylabel("Value's Frequencies")
    plt.title("Class Column's Bar Chart")


def Lab04_Task9_2018_3_60_014():    
    print('\nV4 Skeweness:',round(df['V4'].skew(),5))
    print('\033[1;33;1mV4 is positive skewed.\033[0m')
    df.hist(column=['V4'],bins=100, color='lightgreen', edgecolor='black', linewidth=0.5)
    
    print('\nV3 Skeweness:',round(df['V3'].skew(),5))
    print('\033[1;33;1mV3 is negative skewed.\033[0m')
    df.hist(column=['V3'],bins=100, color='darkred', edgecolor='black', linewidth=0.5)

    print('\nV9 kurtosis:',round(df['V9'].kurt(),5))
    print('\033[1;33;1mV9 is Leptokurtic kurtosis.\033[0m')
    df.hist(column=['V9'],bins=100, color='lightgreen', edgecolor='black', linewidth=0.5)

    print('\nTime kurtosis:',round(df['Time'].kurt(),5))
    print('\033[1;33;1mTime is Platykurtic kurtosis.\033[0m')
    df.hist(column=['Time'],bins=100, color='darkred', edgecolor='black', linewidth=0.5)

    
def Lab04_Task10_2018_3_60_014():
    print('\n\033[1;33;1mHighest Positive Correlation:\033[0m')
    corr = df.corr()
    positive_corr = corr[corr<1]
    highest_positive_corr = positive_corr.max()
    print(highest_positive_corr)
    return highest_positive_corr
    
    
def Lab04_Task11_2018_3_60_014():
    positive_corr = Lab04_Task10_2018_3_60_014()
    df_corr = pd.DataFrame(positive_corr)
    df_corr['index'] = df_corr.index
    df_corr.columns = ['values','index']
    df_corr.plot.scatter(x='index',y='values')
    
    
def Lab04_Task12_2018_3_60_014():
    print('\n\033[1;33;1mHighest Negative Correlation:\033[0m')
    corr = df.corr()
    highest_negative_corr = corr.min()
    print(highest_negative_corr)
    return highest_negative_corr
    

def Lab04_Task13_2018_3_60_014():
    negative_corr = Lab04_Task12_2018_3_60_014()
    df_corr = pd.DataFrame(negative_corr)
    df_corr['index'] = df_corr.index
    df_corr.columns = ['values','index']
    df_corr.plot.scatter(x='index',y='values')
    

def Lab04_Task14_2018_3_60_014():
    df.boxplot(column=['Amount'])
    
    
def Lab04_Task15_2018_3_60_014():
    value0_df = df[['Amount','Class']].query('Class==0')
    value0_series = value0_df['Amount']
    value1_df = df[['Amount','Class']].query('Class==1')
    value1_series = value1_df['Amount']
    cols = [value0_series, value1_series]
    figure, plot= plt.subplots()
    plot.boxplot(cols)

    print('\n\033[1;33;1mAnswer Explanation:\n\033[0m')    
    print('Yes, I found particular pattern by just considering Amount Column. It has high number of outliers.')


#%%

# Task - 1

Lab04_Task1_2018_3_60_014()


#%%

# Task - 2

Lab04_Task2_2018_3_60_014()


#%%

# Task - 3

Lab04_Task3_2018_3_60_014()

        
#%%

# Task - 4

Lab04_Task4_2018_3_60_014()


#%%

# Task - 5

Lab04_Task5_2018_3_60_014()


#%%

# Task - 6

Lab04_Task6_2018_3_60_014()


#%%

# Task - 7

Lab04_Task7_2018_3_60_014()


#%%

# Task - 8

Lab04_Task8_2018_3_60_014()


#%%

# Task - 9

Lab04_Task9_2018_3_60_014()


#%%

# Task - 10

Lab04_Task10_2018_3_60_014()


#%%

# Task - 11

Lab04_Task11_2018_3_60_014()


#%%

# Task - 12

Lab04_Task12_2018_3_60_014()


#%%

# Task - 13

Lab04_Task13_2018_3_60_014()


#%%

# Task - 14

Lab04_Task14_2018_3_60_014()


#%%

# Task - 15

Lab04_Task15_2018_3_60_014()



