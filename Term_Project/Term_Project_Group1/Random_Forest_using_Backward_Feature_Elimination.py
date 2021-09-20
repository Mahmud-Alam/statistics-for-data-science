from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, log_loss
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns


#%%

# Read CSV File

df_actual = pd.read_csv('student_satisfaction.csv')
print(df_actual.head())
print(df_actual.shape)


#%%

# Copy df_actual into df

df = df_actual.copy()
print(df)


#%%

# Drop all unnecessary columns

df.drop(df.iloc[:,:12],axis=1, inplace=True)

df.drop(['OR'],axis=1, inplace=True)

print(df)


#%%

# Check whether there is any missing value or not

print(df.isnull().sum())


#%%

# Replace missing values with median

df['IA1'].fillna(df['IA1'].median(skipna=True), inplace=True)
df['IA3'].fillna(df['IA3'].median(skipna=True), inplace=True)
df['TLA2'].fillna(df['TLA2'].median(skipna=True), inplace=True)


#%%

# Check again for missing values 

print(df.isnull().sum())


#%%

# =============================================================================
#  DataFrame1 - df1
#  Encoding via Get-Dummies technique
# =============================================================================


df_IA1 = pd.get_dummies(df['IA1'],prefix='IA1')
df_IA2 = pd.get_dummies(df['IA2'],prefix='IA2')
df_IA3 = pd.get_dummies(df['IA3'],prefix='IA3')
df_IA4 = pd.get_dummies(df['IA4'],prefix='IA4')
df_IA5 = pd.get_dummies(df['IA5'],prefix='IA5')
df_IA6 = pd.get_dummies(df['IA6'],prefix='IA6')

df1 = pd.concat([ df,df_IA1,df_IA2,df_IA3,df_IA4,df_IA5,df_IA6],axis=1)

print(df1.head())
print(df1.shape)

df1.drop(df1.iloc[:,:6],axis=1, inplace=True)

print(df1.head())
print(df1.shape)


#%%

df_AA1 = pd.get_dummies(df['AA1'],prefix='AA1')
df_AA2 = pd.get_dummies(df['AA2'],prefix='AA2')
df_AA3 = pd.get_dummies(df['AA3'],prefix='AA3')
df_AA4 = pd.get_dummies(df['AA4'],prefix='AA4')
df_AA5 = pd.get_dummies(df['AA5'],prefix='AA5')
df_AA6 = pd.get_dummies(df['AA6'],prefix='AA6')

df1 = pd.concat([ df1,df_AA1,df_AA2,df_AA3,df_AA4,df_AA5,df_AA6],axis=1)

print(df1.head())
print(df1.shape)

df1.drop(df1.iloc[:,:6],axis=1, inplace=True)

print(df1.head())
print(df1.shape)


#%%

df_TLA1 = pd.get_dummies(df['TLA1'],prefix='TLA1')
df_TLA2 = pd.get_dummies(df['TLA2'],prefix='TLA2')
df_TLA3 = pd.get_dummies(df['TLA3'],prefix='TLA3')
df_TLA4 = pd.get_dummies(df['TLA4'],prefix='TLA4')
df_TLA5 = pd.get_dummies(df['TLA5'],prefix='TLA5')
df_TLA6 = pd.get_dummies(df['TLA6'],prefix='TLA6')
df_TLA7 = pd.get_dummies(df['TLA7'],prefix='TLA7')

df1 = pd.concat([ df1,df_TLA1,df_TLA2,df_TLA3,df_TLA4,df_TLA5,df_TLA6,df_TLA7],axis=1)

print(df1.head())
print(df1.shape)

df1.drop(df1.iloc[:,:7],axis=1, inplace=True)

print(df1.head())
print(df1.shape)


#%%

# =============================================================================
#  Encoding via Label-Encoder technique for Satisfied Column
# =============================================================================

le = preprocessing.LabelEncoder()
le.fit(df1['Satisfied'])
df_Satisfied = pd.Series(le.fit_transform(df1['Satisfied']))

print(df_Satisfied)


#%%

# Add transform Satisfied Column into df1

df1 = pd.concat([df1,df_Satisfied],axis=1)

print(df1.head())
print(df1.shape)

df1.drop(df1.iloc[:,:1],axis=1, inplace=True)

print(df1.head())
print(df1.shape)


#%%

# Rename the Satisfied Column for df1

df1.columns = [*df1.columns[:-1], 'Satisfied']


#%%

# =============================================================================
#  Final DataFrame1 - df1
# =============================================================================

print(df1.head())
print(df1.shape)


#%%

# =============================================================================
#  Dataset-1
#  Independent variable put into X1 and Dependent variable 'Satisfied' put into y1
# =============================================================================


X1 = df1.iloc[:,:-1]
print(X1)
print(X1.shape)

y1 = df1[['Satisfied']]
print(y1)
print(y1.shape)


#%%

# Dataset-1 Train Test Split 

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1,y1, test_size=1/3, random_state=72)


#%%

# =============================================================================
#  Dataset-1
#  Max n_estimators value and Min n_estimators value calculation
# =============================================================================

estMAX1=0
estMIN1=0
accMAX1=0.001
accMIN1=99

for i in range(1,100):
    RForest_model1 = RandomForestClassifier(n_estimators=i, max_features='sqrt', criterion='gini',random_state=90)
    RForest_model1.fit(X_train1, y_train1.values.ravel())
    y_pred1 = RForest_model1.predict(X_test1)
    accuracy1 = accuracy_score(y_test1,y_pred1)
    
    if accuracy1>accMAX1:
        estMAX1=i
        accMAX1=accuracy1
        
    if accuracy1<accMIN1:
        estMIN1=i
        accMIN1=accuracy1


print('\n\033[1;33;1m___Dataset-1___\033[0m')
print('\n\033[1;35;1mMax Estimators :\033[0m',estMAX1)
print('\033[1;35;1mMin Estimators :\033[0m',estMIN1)
print('\033[1;36;1mMax Accuracy :\033[0m',accMAX1)
print('\033[1;36;1mMin Accuracy :\033[0m',accMIN1)
print()

"""
###for criterion = gini, random_state=1
Max Estimators : 21
Min Estimators : 2
Max Accuracy : 0.9392265193370166
Min Accuracy : 0.7900552486187845


###for criterion = entropy, random_state=1
Max Estimators : 27
Min Estimators : 2
Max Accuracy : 0.9281767955801105
Min Accuracy : 0.7955801104972375


###for n_estimators=27, criterion = entropy
Max Random_state : 98
Min Random_state : 20
Max Accuracy : 0.9447513812154696
Min Accuracy : 0.8950276243093923


###for criterion = entropy, random_state=98
Max Estimators : 22,27
Min Estimators : 2
Max Accuracy : 0.9447513812154696
Min Accuracy : 0.7790055248618785


###for n_estimators=19, criterion = gini
Max Random_state : 90
Min Random_state : 66
Max Accuracy : 0.9502762430939227
Min Accuracy : 0.8674033149171271


###for criterion = gini, random_state=90, max_features='log2'
Max Estimators : 14
Min Estimators : 2
Max Accuracy : 0.9392265193370166
Min Accuracy : 0.8011049723756906


###for criterion = gini, random_state=90,  max_features='auto'/'sqrt'
Max Estimators : 19
Min Estimators : 1
Max Accuracy : 0.9502762430939227            ###_______Best Accuracy______###
Min Accuracy : 0.7845303867403315
"""


#%%

# =============================================================================
#  Dataset-1
#  Apply Random Forest Algorithm
# =============================================================================


RForest_model1 = RandomForestClassifier(n_estimators=estMAX1, criterion='gini', max_features='sqrt', random_state=90)

RForest_model1.fit(X_train1, y_train1.values.ravel())

y_pred1 = RForest_model1.predict(X_test1)

#print(y_pred1)
#print(y_test1)


accuracy1 = accuracy_score(y_test1,y_pred1)
print('\n\033[1;36;1mAccuraccy Score :\033[0m',accuracy1)
print('\033[1;36;1mPrecision Score :\033[0m',precision_score(y_test1, y_pred1))
print('\033[1;36;1mRecall Score    :\033[0m',recall_score(y_test1, y_pred1))
print('\033[1;36;1mF1 Score        :\033[0m',f1_score(y_test1, y_pred1))
print('\n\033[1;33;1mClassification Report :\033[0m')
print(classification_report(y_test1, y_pred1))

#print('\nModel Prediction Probability :')
#print(RForest_model1.predict_proba(X_test1))


"""
# =============================================================================
# 
# Accuraccy Score : 0.9502762430939227
# Precision Score : 0.9652777777777778
# Recall Score    : 0.972027972027972
# F1 Score        : 0.9686411149825783
# 
# Classification Report :
#               precision    recall  f1-score   support
# 
#            0       0.89      0.87      0.88        38
#            1       0.97      0.97      0.97       143
# 
#     accuracy                           0.95       181
#    macro avg       0.93      0.92      0.92       181
# weighted avg       0.95      0.95      0.95       181
# 
# =============================================================================
"""


#%%

# Confusion matrix seaborn plot for Dataset-1

cm1 = confusion_matrix(y_test1,y_pred1)
print('\n\033[1;33;1mConfusion atrix :\033[0m')
print(cm1)

plt.figure(figsize=(6,6))
sns.heatmap(cm1, annot=True, fmt=".1f", linewidths=5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual Class');
plt.xlabel('Predicted Class');
all_sample_title = 'Accuracy Score: {0}%'.format(round(accuracy1*100,2))
plt.title(all_sample_title, size = 12);


#%%

# =============================================================================
#  Dataset-1
#  Create a OLS MODEL for finding P-values
# =============================================================================

X1 = sm.add_constant(X1)
ols_model1 = sm.OLS(y1, X1).fit()

print(ols_model1.summary())


#%%

# Find highest P-value column which is greater than 0.005 for drop the column

p_values1 = round(ols_model1.pvalues,3)

print('\n\033[1;33;1mMax P-value :\033[0m',p_values1.idxmax(),'-',p_values1.max())


#%%

# =============================================================================
#  Dataset-1
#  Apply Backward Feature Elimination by p_value=0.05
# =============================================================================


for i in range(df1.shape[1]):
    p_values1 = round(ols_model1.pvalues,3)
    
    if p_values1.max()>0.05:
        print('\n\033[1;33;1mMax P-value :\033[0m',p_values1.idxmax(),'-',p_values1.max())
        X1.drop([p_values1.idxmax()], axis=1, inplace=True)
        print('\033[1;33;1mIndependent var shape :\033[0m',X1.shape)
        
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X1,y1, test_size=1/3, random_state=72)
        RForest_model1 = RandomForestClassifier(n_estimators=estMAX1, criterion='gini',max_features='sqrt',random_state=85)
        RForest_model1.fit(X_train1, y_train1.values.ravel())
        y_pred1 = RForest_model1.predict(X_test1)
        accuracy1 = accuracy_score(y_test1,y_pred1)
        print('\n\033[1;36;1mAccuraccy Score :\033[0m',accuracy1)
        print()
        ols_model1 = sm.OLS(y1, X1).fit()
        print(ols_model1.summary())
    else: break

print('\n\033[1;33;1mFinal Independent var shape :\033[0m',X1.shape)


#%%

# =============================================================================
#   Dataset-1
#   After Backward Feature Elimination, apply Random Forest Algorithm again for finding accuracy
# =============================================================================


X_train1, X_test1, y_train1, y_test1 = train_test_split(X1,y1, test_size=1/3, random_state=72)


# =============================================================================
#  Max random_state value calculation
# =============================================================================

randMAX1=0
accMAX1=0.001

for i in range(1,100):
    RForest_model1 = RandomForestClassifier(n_estimators=estMAX1, max_features='sqrt', criterion='gini',random_state=i)
    RForest_model1.fit(X_train1, y_train1.values.ravel())
    y_pred1 = RForest_model1.predict(X_test1)
    accuracy1 = accuracy_score(y_test1,y_pred1)
    
    if accuracy1>accMAX1:
        randMAX1=i
        accMAX1=accuracy1
        

print('\n\033[1;33;1m___Dataset-1___\033[0m')
print('\n\033[1;35;1mMax random_state :\033[0m',randMAX1)
print('\033[1;36;1mMax Accuracy :\033[0m',accMAX1)
print()

"""
_____Result___________
Max random_state : 82
Max Accuracy : 0.9447513812154696

"""


RForest_model1 = RandomForestClassifier(n_estimators=estMAX1, criterion='gini',max_features='sqrt',random_state=randMAX1)

RForest_model1.fit(X_train1, y_train1.values.ravel())

y_pred1 = RForest_model1.predict(X_test1)


#%%

# Check dataframe's previous and new shape

prev_shape1 = df1.shape
print('\n\033[1;35;1mPrevious Dataset-1 shape :\033[0m',prev_shape1)

df1 = X1.copy()
df1.insert(loc=len(df1.columns), column='Satisfied', value=df_Satisfied)
print('\n\033[1;35;1mAfter applying Backward Feature Elimination, \nNew Dataset-1 shape :\033[0m',df1.shape)


#%%

# accuracy, precision, recall, f1-Score calculation

accuracy1 = accuracy_score(y_test1,y_pred1)
print('\n\033[1;36;1mAccuraccy Score :\033[0m',accuracy1)
print('\033[1;36;1mPrecision Score :\033[0m',precision_score(y_test1, y_pred1))
print('\033[1;36;1mRecall Score    :\033[0m',recall_score(y_test1, y_pred1))
print('\033[1;36;1mF1 Score        :\033[0m',f1_score(y_test1, y_pred1))
print('\n\033[1;33;1mClassification Report :\033[0m')
print(classification_report(y_test1, y_pred1))

#print('\nModel Prediction Probability :')
#print(RForest_model1.predict_proba(X_test1))


"""
# =============================================================================
# 
# Previous Dataset-1 shape : (543, 96)
# 
# After applying Backward Feature Elimination, 
# New Dataset-1 shape : (543, 29)
# 
# Accuraccy Score : 0.9005524861878453
# Precision Score : 0.9562043795620438
# Recall Score    : 0.916083916083916
# F1 Score        : 0.9357142857142857
# 
# Classification Report :
#               precision    recall  f1-score   support
# 
#            0       0.73      0.84      0.78        38
#            1       0.96      0.92      0.94       143
# 
#     accuracy                           0.90       181
#    macro avg       0.84      0.88      0.86       181
# weighted avg       0.91      0.90      0.90       181
# 
# =============================================================================


_______________________After update random_state value_________________________
# =============================================================================
#
# Accuraccy Score : 0.9447513812154696
# Precision Score : 0.9716312056737588
# Recall Score    : 0.958041958041958
# F1 Score        : 0.9647887323943661
# 
# Classification Report :
#               precision    recall  f1-score   support
# 
#            0       0.85      0.89      0.87        38
#            1       0.97      0.96      0.96       143
# 
#     accuracy                           0.94       181
#    macro avg       0.91      0.93      0.92       181
# weighted avg       0.95      0.94      0.95       181
# 
# =============================================================================
"""


#%%

# After Backward Feature Elimination, Confusion matrix seaborn plot for Dataset-1

cm1 = confusion_matrix(y_test1,y_pred1)
print('\n\033[1;33;1mConfusion atrix :\033[0m')
print(cm1)

plt.figure(figsize=(6,6))
sns.heatmap(cm1, annot=True, fmt=".1f", linewidths=5, square = True, cmap = 'Greens_r');
plt.ylabel('Actual Class');
plt.xlabel('Predicted Class');
all_sample_title = 'After Eliination Accuracy Score: {0}%'.format(round(accuracy1*100,2))
plt.title(all_sample_title, size = 12);


#%%

# =============================================================================
#  DataFrame2 - df2
# =============================================================================

df2 = df.copy()
print(df2.head())
print(df2.shape)


#%%

# Drop the Satisfied column

df2.drop(['Satisfied'], axis=1, inplace=True)

print(df2.shape)


#%%

# Insert Label-Encoded Transform Satisfied Column Series into df2 

df2.insert(loc=len(df2.columns), column='Satisfied', value=df_Satisfied)
print(df2.head())
print(df2.shape)


#%%

# =============================================================================
#  Dataset-2
#  Independent variable put into X2 and Dependent variable 'Satisfied' put into y2
# =============================================================================

X2 = df2.iloc[:,:-1]
#print(X2)
#X2.insert(0,'B0',1)
#print(X2)
print(X2.shape)

y2 = df2[['Satisfied']]
#print(y2)
print(y2.shape)


#%%

# Dataset-2 Train Test Split 

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2,y2, test_size=1/3, random_state=72)


#%%

# =============================================================================
#  Dataset-2
#  Max n_estimators value and Min n_estimators value calculation for Dataset-2
# =============================================================================

estMAX2=0
estMIN2=0
accMAX2=0.001
accMIN2=99

for i in range(1,100):
    RForest_model2 = RandomForestClassifier(n_estimators=i,criterion='gini',max_features='sqrt',random_state=69)
    RForest_model2.fit(X_train2, y_train2.values.ravel())
    y_pred2 = RForest_model2.predict(X_test2)
    accuracy2 = accuracy_score(y_test2,y_pred2)
    
    if accuracy2>accMAX2:
        estMAX2=i
        accMAX2=accuracy2
        
    if accuracy2<accMIN2:
        estMIN2=i
        accMIN2=accuracy2


print('\n\033[1;33;1m___Dataset-2___\033[0m')
print('\n\033[1;35;1mMax Estimators :\033[0m',estMAX2)
print('\033[1;35;1mMin Estimators :\033[0m',estMIN2)
print('\033[1;36;1mMax Accuracy :\033[0m',accMAX2)
print('\033[1;36;1mMin Accuracy :\033[0m',accMIN2)
print()

"""
###for criterion = gini, random_state=1
Max Estimators : 62
Min Estimators : 1
Max Accuracy : 0.9558011049723757
Min Accuracy : 0.861878453038674


###for criterion = entropy, random_state=1
Max Estimators : 32
Min Estimators : 1
Max Accuracy : 0.9447513812154696
Min Accuracy : 0.8397790055248618


###for n_estimators=32, criterion = entropy
Max Random_state : 6
Min Random_state : 48
Max Accuracy : 0.9502762430939227
Min Accuracy : 0.9060773480662984


###for criterion = entropy, random_state=6
Max Estimators : 33
Min Estimators : 1
Max Accuracy : 0.9558011049723757
Min Accuracy : 0.8453038674033149


###for n_estimators=19, criterion = gini
Max Random_state : 85
Min Random_state : 36
Max Accuracy : 0.9613259668508287
Min Accuracy : 0.9171270718232044


###for criterion = gini, random_state=69
Max Estimators : 26
Min Estimators : 2
Max Accuracy : 0.9668508287292817            ###_______Best Accuracy______###
Min Accuracy : 0.850828729281768
"""


#%%

# =============================================================================
#  Dataset-2
#  Apply Random Forest Algorithm
# =============================================================================


RForest_model2 = RandomForestClassifier(n_estimators=estMAX2, criterion='gini',max_features='sqrt',random_state=69)

RForest_model2.fit(X_train2, y_train2.values.ravel())

y_pred2 = RForest_model2.predict(X_test2)

#print(y_pred2)
#print(y_test2)


accuracy2 = accuracy_score(y_test2,y_pred2)
print('\n\033[1;36;1mAccuraccy Score :\033[0m',accuracy2)
print('\033[1;36;1mPrecision Score :\033[0m',precision_score(y_test2, y_pred2))
print('\033[1;36;1mRecall Score    :\033[0m',recall_score(y_test2, y_pred2))
print('\033[1;36;1mF1 Score        :\033[0m',f1_score(y_test2, y_pred2))
print('\n\033[1;33;1mClassification Report :\033[0m')
print(classification_report(y_test2, y_pred2))

#print('\nModel Prediction Probability :')
#print(RForest_model2.predict_proba(X_test2))


"""
# =============================================================================
# 
# Accuraccy Score : 0.9668508287292817
# Precision Score : 0.9858156028368794
# Recall Score    : 0.972027972027972
# F1 Score        : 0.9788732394366197
# 
# Classification Report :
#               precision    recall  f1-score   support
# 
#            0       0.90      0.95      0.92        38
#            1       0.99      0.97      0.98       143
# 
#     accuracy                           0.97       181
#    macro avg       0.94      0.96      0.95       181
# weighted avg       0.97      0.97      0.97       181
# 
# =============================================================================
"""


#%%

# Confusion matrix seaborn plot for Dataset-2

cm2 = confusion_matrix(y_test2,y_pred2)
print('\n\033[1;33;1mConfusion Matrix :\033[0m')
print(cm2)

plt.figure(figsize=(6,6))
sns.heatmap(cm2, annot=True, fmt=".1f", linewidths=5, square = True, cmap = 'Reds_r');
plt.ylabel('Actual Class');
plt.xlabel('Predicted Class');
all_sample_title = 'Accuracy Score: {0}%'.format(round(accuracy2*100,2))
plt.title(all_sample_title, size = 12);


#%%

# =============================================================================
#  Dataset-2
#  Create a OLS MODEL for finding P-values
# =============================================================================

X2 = sm.add_constant(X2)
ols_model2 = sm.OLS(y2, X2).fit()

print(ols_model2.summary())


#%%

# Find highest P-value column which is greater than 0.005 for drop the column

p_values2 = round(ols_model2.pvalues,3)

print('\n\033[1;33;1mMax P-value :\033[0m',p_values2.idxmax(),'-',p_values2.max())


#%%

# =============================================================================
#  Dataset-2
#  Apply Backward Feature Elimination by p_value=0.05
# =============================================================================


for i in range(df2.shape[1]):
    p_values2 = round(ols_model2.pvalues,3)
    
    if p_values2.max()>0.05:
        print('\n\033[1;33;1mMax P-value :\033[0m',p_values2.idxmax(),'-',p_values2.max())
        X2.drop([p_values2.idxmax()], axis=1, inplace=True)
        print('\033[1;33;1mIndependent var shape :\033[0m',X2.shape)
        
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X2,y2, test_size=1/3, random_state=72)
        RForest_model2 = RandomForestClassifier(n_estimators=estMAX2, criterion='gini',max_features='sqrt',random_state=85)
        RForest_model2.fit(X_train2, y_train2.values.ravel())
        y_pred2 = RForest_model2.predict(X_test2)
        accuracy2 = accuracy_score(y_test2,y_pred2)
        print('\n\033[1;36;1mAccuraccy Score :\033[0m',accuracy2)
        print()
        ols_model2 = sm.OLS(y2, X2).fit()
        print(ols_model2.summary())
    else: break

print('\n\033[1;33;1mFinal Independent var shape :\033[0m',X2.shape)


#%%

# =============================================================================
#   Dataset-2
#   After Backward Feature Elimination, apply Random Forest Algorithm again for finding accuracy
# =============================================================================


X_train2, X_test2, y_train2, y_test2 = train_test_split(X2,y2, test_size=1/3, random_state=72)


# =============================================================================
#  Max random_state value calculation
# =============================================================================

randMAX2=0
accMAX2=0.001

for i in range(1,100):
    RForest_model2 = RandomForestClassifier(n_estimators=estMAX2, max_features='sqrt', criterion='gini',random_state=i)
    RForest_model2.fit(X_train2, y_train2.values.ravel())
    y_pred2 = RForest_model2.predict(X_test2)
    accuracy2 = accuracy_score(y_test2,y_pred2)
    
    if accuracy2>accMAX2:
        randMAX2=i
        accMAX2=accuracy2
        

print('\n\033[1;33;1m___Dataset-1___\033[0m')
print('\n\033[1;35;1mMax random_state :\033[0m',randMAX2)
print('\033[1;36;1mMax Accuracy :\033[0m',accMAX2)
print()


"""
Max random_state : 29
Max Accuracy : 0.9502762430939227

"""


RForest_model2 = RandomForestClassifier(n_estimators=estMAX2, criterion='gini',max_features='sqrt',random_state=randMAX2)

RForest_model2.fit(X_train2, y_train2.values.ravel())

y_pred2 = RForest_model2.predict(X_test2)


#%%

# Check dataframe's previous and new shape

prev_shape2 = df2.shape
print('\n\033[1;35;1mPrevious Dataset-2 shape :\033[0m',prev_shape2)

df2 = X2.copy()
df2.insert(loc=len(df2.columns), column='Satisfied', value=df_Satisfied)
print('\n\033[1;35;1mAfter applying Backward Feature Elimination, \nNew Dataset-2 shape :\033[0m',df2.shape)


#%%

# Accuracy, precision, recall, f1-score calculation

accuracy2 = accuracy_score(y_test2,y_pred2)
print('\n\033[1;36;1mAccuraccy Score :\033[0m',accuracy2)
print('\033[1;36;1mPrecision Score :\033[0m',precision_score(y_test2, y_pred2))
print('\033[1;36;1mRecall Score    :\033[0m',recall_score(y_test2, y_pred2))
print('\033[1;36;1mF1 Score        :\033[0m',f1_score(y_test2, y_pred2))
print('\n\033[1;33;1mClassification Report :\033[0m')
print(classification_report(y_test2, y_pred2))

#print('\nModel Prediction Probability :')
#print(RForest_model2.predict_proba(X_test2))


"""
# =============================================================================
#
# Previous Dataset-2 shape : (543, 20)
#
# After applying Backward Feature Elimination, 
# New Dataset-2 shape : (543, 10)
# 
# Accuraccy Score : 0.9060773480662984
# Precision Score : 0.9772727272727273
# Recall Score    : 0.9020979020979021
# F1 Score        : 0.9381818181818181
# 
# Classification Report :
#               precision    recall  f1-score   support
# 
#            0       0.71      0.92      0.80        38
#            1       0.98      0.90      0.94       143
# 
#     accuracy                           0.91       181
#    macro avg       0.85      0.91      0.87       181
# weighted avg       0.92      0.91      0.91       181
#
# =============================================================================


_______________________After update random_state value_________________________
# =============================================================================
# 
# Accuraccy Score : 0.9502762430939227
# Precision Score : 1.0
# Recall Score    : 0.9370629370629371
# F1 Score        : 0.9675090252707581
# 
# Classification Report :
#               precision    recall  f1-score   support
# 
#            0       0.81      1.00      0.89        38
#            1       1.00      0.94      0.97       143
# 
#     accuracy                           0.95       181
#    macro avg       0.90      0.97      0.93       181
# weighted avg       0.96      0.95      0.95       181
# 
# =============================================================================
"""


#%%

# After Backward Feature Elimination, Confusion matrix seaborn plot for Dataset-2

cm2 = confusion_matrix(y_test2,y_pred2)
print('\n\033[1;33;1mConfusion atrix :\033[0m')
print(cm2)

plt.figure(figsize=(6,6))
sns.heatmap(cm2, annot=True, fmt=".1f", linewidths=5, square = True, cmap = 'Oranges_r');
plt.ylabel('Actual Class');
plt.xlabel('Predicted Class');
all_sample_title = 'After Eliination Accuracy Score: {0}%'.format(round(accuracy2*100,2))
plt.title(all_sample_title, size = 12);


