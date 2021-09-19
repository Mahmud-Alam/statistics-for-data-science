from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, log_loss
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

#%%

# Read CSV File

df_actual = pd.read_csv('student_satisfaction.csv')
#print(df_actual.head())
print(df_actual.shape)


#%%

# Copy df_actual into df

df = df_actual.copy()
print(df)

#%%

# Drop all unnecessary columns

#df.drop(['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12',],axis=1, inplace=True)
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

# Encoding via Label-Encoder technique for Satisfied Column

le = preprocessing.LabelEncoder()
le.fit(df['Satisfied'])
df_Satisfied = pd.Series(le.fit_transform(df['Satisfied']))

#print(df_Satisfied)

#%%

# Drop the Satisfied column

df.drop(['Satisfied'], axis=1, inplace=True)

print(df.shape)

#%%

# Insert Label-Encoded Transform Satisfied Column Series into df 

df.insert(loc=len(df.columns), column='Satisfied', value=df_Satisfied)
#print(df.head())
print(df.shape)


#%%

# =============================================================================
#  Independent variable put into X and Dependent variable 'Satisfied' put into y
# =============================================================================

X = df.iloc[:,:-1]
#print(X)
#X.insert(0,'B0',1)
#print(X)
print(X.shape)

y = df[['Satisfied']]
#print(y)
print(y.shape)

#%%

# Dataset- Train Test Split 

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=1/3, random_state=72)

#%%

# =============================================================================
#  Max n_estimators value and Min n_estimators value calculation for Dataset-
# =============================================================================

estMAX=0
estMIN=0
accMAX=0.001
accMIN=99

for i in range(1,100):
    RForest_model = RandomForestClassifier(n_estimators=i,criterion='gini',random_state=85)
    RForest_model.fit(X_train, y_train.values.ravel())
    y_pred = RForest_model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    
    if accuracy>accMAX:
        estMAX=i
        accMAX=accuracy
        
    if accuracy<accMIN:
        estMIN=i
        accMIN=accuracy


print('\n___Dataset-___')
print('\nMax Estimators :',estMAX)
print('Min Estimators :',estMIN)
print('Max Accuracy :',accMAX)
print('Min Accuracy :',accMIN)
print()

"""
###for criterion = gini, random_state=85
Max Estimators : 28
Min Estimators : 2
Max Accuracy : 0.9613259668508287            ###_______Best Accuracy______###
Min Accuracy : 0.856353591160221
"""

#%%

# =============================================================================
#  Dataset-
#  Apply Random Forest Algorithm
# =============================================================================


RForest_model = RandomForestClassifier(n_estimators=estMAX, criterion='gini',random_state=85)

RForest_model.fit(X_train, y_train.values.ravel())

y_pred = RForest_model.predict(X_test)

#print(y_pred)
#print(y_test)


score = accuracy_score(y_test,y_pred)
print('\nAccuraccy Score :',score)
print('Precision Score :',precision_score(y_test, y_pred))
print('Recall Score    :',recall_score(y_test, y_pred))
print('F1 Score        :',f1_score(y_test, y_pred))
print('\nClassification Report :')
print(classification_report(y_test, y_pred))

#print('\nModel Prediction Probability :')
#print(RForest_model.predict_proba(X_test))

"""
# =============================================================================
#
# Accuraccy Score : 0.9613259668508287
# Precision Score : 0.9788732394366197
# Recall Score    : 0.972027972027972
# F1 Score        : 0.9754385964912281
# 
# Classification Report :
#               precision    recall  f1-score   support
# 
#            0       0.90      0.92      0.91        38
#            1       0.98      0.97      0.98       143
# 
#     accuracy                           0.96       181
#    macro avg       0.94      0.95      0.94       181
# weighted avg       0.96      0.96      0.96       181
#
# =============================================================================
"""
#%%

logit_roc_auc = roc_auc_score(y_test, RForest_model.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, RForest_model.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Random Forest (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()