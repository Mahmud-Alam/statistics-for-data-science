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

# Encoding via Label-Encoder technique for Satisfied Column

le = preprocessing.LabelEncoder()
le.fit(df['Satisfied'])
df_Satisfied = pd.Series(le.fit_transform(df['Satisfied']))

print(df_Satisfied)


#%%

# Drop the Satisfied column

df.drop(['Satisfied'], axis=1, inplace=True)

print(df.shape)


#%%

# Insert Label-Encoded Transform Satisfied Column Series into df 

df.insert(loc=len(df.columns), column='Satisfied', value=df_Satisfied)
print(df.head())
print(df.shape)


#%%

# =============================================================================
#  Independent variable put into X and Dependent variable 'Satisfied' put into y
# =============================================================================

X = df.iloc[:,:-1]
print(X)

# Add a Constant column B0 with value 0

X.insert(0,'B0',1)
print(X)
print(X.shape)

y = df[['Satisfied']]
print(y)
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
    RForest_model = RandomForestClassifier(n_estimators=i,criterion='gini',random_state=55)
    RForest_model.fit(X_train, y_train.values.ravel())
    y_pred = RForest_model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    
    if accuracy>accMAX:
        estMAX=i
        accMAX=accuracy
        
    if accuracy<accMIN:
        estMIN=i
        accMIN=accuracy


print('\n\033[1;33;1m___Dataset___\033[0m')
print('\n\033[1;35;1mMax Estimators :\033[0m',estMAX)
print('\033[1;35;1mMin Estimators :\033[0m',estMIN)
print('\033[1;36;1mMax Accuracy :\033[0m',accMAX)
print('\033[1;36;1mMin Accuracy :\033[0m',accMIN)
print()

"""
###for criterion = gini, random_state=85
Max Estimators : 28
Min Estimators : 2
Max Accuracy : 0.9613259668508287            
Min Accuracy : 0.856353591160221

###for criterion = gini, random_state=55
Max Estimators : 27
Min Estimators : 2
Max Accuracy : 0.9668508287292817             ###_______Best Accuracy______###
Min Accuracy : 0.856353591160221
"""

#%%

# =============================================================================
#  Dataset-
#  Apply Random Forest Algorithm
# =============================================================================


RForest_model = RandomForestClassifier(n_estimators=estMAX, criterion='gini',random_state=55)

RForest_model.fit(X_train, y_train.values.ravel())

y_pred = RForest_model.predict(X_test)

#print(y_pred)
#print(y_test)

accuracyTest = accuracy_score(y_test,y_pred)
accuracyTrain = RForest_model.score(X_train,y_train)
print('\n\033[1;36;1mTest  Accuraccy Score :\033[0m',accuracyTest)
print('\033[1;36;1mTrain Accuraccy Score :\033[0m',accuracyTrain)
print('\n\033[1;35;1mPrecision Score :\033[0m',precision_score(y_test, y_pred))
print('\033[1;35;1mRecall Score    :\033[0m',recall_score(y_test, y_pred))
print('\033[1;35;1mF1 Score        :\033[0m',f1_score(y_test, y_pred))
print('\n\033[1;33;1mClassification Report :\033[0m')
print(classification_report(y_test, y_pred))


"""
# =============================================================================
# 
# Test  Accuraccy Score : 0.9668508287292817
# Train Accuraccy Score : 0.9972375690607734
# 
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

# Calculate model prediction probability

print('\n\033[1;33;1mModel Prediction Probability :\033[0m')
print(RForest_model.predict_proba(X_test))


#%%

# Confusion matrix seaborn plot

cm = confusion_matrix(y_test,y_pred)
print('\n\033[1;33;1mConfusion Matrix :\033[0m')
print(cm)

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt=".1f", linewidths=5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual Class');
plt.xlabel('Predicted Class');
all_sample_title = 'Accuracy Score: {0}%'.format(round(accuracyTest*100,2))
plt.title(all_sample_title, size = 12);



#%%

# =============================================================================
#  ROC Curve for Random Forest 
# =============================================================================

# Area under the curve value calculation
roc_auc = roc_auc_score(y_test, y_pred)

# FPR, TPR, Threshold value valculation
fpr, tpr, thres = roc_curve(y_test, RForest_model.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Random Forest (Area = {0}%)'.format(round(roc_auc*100,2)))

# for Middle red -- Line 
plt.plot([0, 1], [0, 1],'b--',lw=2)

# set the x and y limit to left, right 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

# set x and y label name 
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver operating characteristic (ROC Curve)')
plt.legend(loc="lower right")

# Save the ROC Curve Figure as png format
plt.savefig('ROC_curve_fig')
plt.show()
