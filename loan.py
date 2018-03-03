# -*- coding: utf-8 -*-
"""
Created on Sun May 28 17:27:05 2017

@author: rikin
"""

import numpy as np
import pandas as pd
import sys
import os 
from pylab import * 

#Import Dataset
df= pd.read_csv('loan.csv')

#Clearing out preliminary variables
df.drop(['id', 'member_id', 'emp_title'], axis=1, inplace=True)
df.replace('n/a', np.nan,inplace=True)
df.emp_length.fillna(value=0,inplace=True)
df.drop(['pymnt_plan','url','desc','title' ],1, inplace=True)

df['emp_length'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
df['emp_length'] = df['emp_length'].astype(int)

df['term'] = df['term'].apply(lambda x: x.lstrip())

#Changing data time format
import datetime

df.issue_d.fillna(value=np.nan,inplace=True)
issue_d_todate = pd.to_datetime(df.issue_d)
df.issue_d = pd.Series(df.issue_d).str.replace('-2015', '')
earliest_cr_line = pd.to_datetime(df.earliest_cr_line)

credit_line_year=[]
for i in df['earliest_cr_line']:
    credit_line_year.append(str(i)[4:])


df=pd.concat([df,pd.DataFrame(credit_line_year)],axis=1)

#Visualizing important variables
import seaborn as sns
import matplotlib 


#Visualizing Loan Status by loan amount
snstemp=pd.concat([df.loan_status,df.grade],axis=1)
df1 = snstemp[snstemp.loan_status == 0]
df2 = snstemp[snstemp.loan_status == 1]

df1=df1.sample(n=207723, axis=0)

snstemp=pd.concat([df1,df2],axis=0)
ax=sns.countplot(x=snstemp.loan_status,hue=snstemp.grade,hue_order=['A','B','C','D','E', 'F','G'],palette="Set3")
ax.set(xlabel='Loan Status',ylabel='Number of Customers',title='Loan Status v Loan Grade (Base Interest Rate)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()




#Prepare Final Data Set for Classification
clean_data=df[['loan_status', 'loan_amnt', 'emp_length', 'verification_status', 
         'home_ownership', 'annual_inc', 'purpose', 'inq_last_6mths', 
         'open_acc', 'pub_rec', 'revol_util', 'dti', 'total_acc', 'delinq_2yrs','mths_since_last_delinq']]

df['loan_status']=(df['loan_status'].values =='Fully Paid').astype(int)

#Introduce Dummy Variables for Categorical Features
dummy_home=pd.get_dummies(clean_data['home_ownership'])
dummy_home=dummy_home[['MORTGAGE','OTHER','OWN','RENT']]
clean_data = pd.concat([clean_data, dummy_home], axis=1)

dummy_verified=pd.get_dummies(clean_data['verification_status'])
clean_data = pd.concat([clean_data, dummy_verified], axis=1)

df=pd.concat([df,clean_data.loan_status],axis=1)

#issued = pd.to_datetime((df.issue_d))
#timehistory=issue_d_todate-earliest_cr_line
a=pd.DataFrame(credit_line_year)
clean_data = pd.concat([clean_data, a], axis=1)

# Encode high cardinality feature with their probability   
def featureEncoder(a, high_card, cutoff = 10):
   table = a[high_card].value_counts().to_frame()
   new_column = a.groupby(high_card)['loan_status'].sum()
   table['prior']=new_column.reset_index(drop=True)
   finaltable = table[table[high_card]>cutoff]
   finaltablemiss = table[table[high_card]<=cutoff]
   finaltable[high_card+'_trans'] = finaltable['prior']/finaltable[high_card]
   try:
       finaltablemiss[high_card+'_trans'] = float(sum(finaltablemiss['prior']))/float(sum(finaltablemiss[high_card]))
   except:
       pass
   finaltable = finaltable.append(finaltablemiss).drop(labels=[high_card,'prior'], axis=1)
   return finaltable

# Create transformed feature, named transformer_1, from high cardinality feature
transformer_1 =pd.DataFrame()
# Set up CV so that outcome var won't leak into the feature, fit to train, create for test
from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)
for dev_index, val_index in kf.split(np.zeros(len(clean_data)), clean_data['loan_status']):
   finaltable = featureEncoder(clean_data.iloc[dev_index], 'purpose', 5)
   chunk = pd.merge(left=clean_data.iloc[val_index], right = finaltable, how = 'left',
                    left_on = 'purpose',
                    right_index=True)
   transformer_1 = transformer_1.append(chunk)

clean_data=transformer_1
clean_data=clean_data.drop(clean_data.columns[[3, 4, 6]], axis=1)

mths=clean_data['mths_since_last_delinq']
mths=np.array(mths)
for i in range(0,len(mths)):
    if(np.isnan(mths[i])==True):
        mths[i]=0
    else:
        mths[i]=1
mths_since_last_delq=pd.DataFrame(mths)
clean_data = pd.concat([clean_data, mths_since_last_delq], axis=1)
clean_data=clean_data.drop(clean_data.columns[[19]], axis=1) 
clean_data = clean_data.rename(columns={'purpose_trans': 'purpose'})

df1 = clean_data[clean_data.loan_status == 0]
df2 = clean_data[clean_data.loan_status == 1]

df1=df1.sample(n=207723, axis=0)

clean_data=pd.concat([df1,df2],axis=0)     

   
temp_data=clean_data[[0,1,4,5,8,9,12,14,15,17]]
#Define indepdent and dependent variables
X = pd.DataFrame(temp_data.iloc[:, 1:].values)
y = pd.DataFrame(temp_data.iloc[:, 0].values)

#Fill in missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)

#Outlier Analysis
from scipy.spatial.distance import mahalanobis
import scipy as sp



xoutliers = pd.DataFrame(X)

Sx = xoutliers.cov().values
Sx = sp.linalg.inv(Sx)

mean = xoutliers.mean().values

def mahalanobisR(xoutliers,meanCol,IC):
    m = []
    for i in range(xoutliers.shape[0]):
        m.append(mahalanobis(xoutliers.ix[i,:],meanCol,IC) ** 2)
    return(m)


#Split Training and Test Set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Fit Training Set
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


X_train_perm=X_train
X_test_perm=X_test
y_train_perm=y_train
y_test_perm=y_test


#SVM Classification
from sklearn import svm
clf = svm.SVC(kernel = 'linear', random_state = 0)
clf.fit(X_train_perm,y_train_perm)
# Predicting the Test set results
y_pred = clf.predict(X_test_perm)

#Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train_perm, y_train_perm)

y_pred = pd.DataFrame(classifier.predict(X_test_perm))

#Gradient Boost Classifier
from sklearn.ensemble import GradientBoostingClassifier as rf
crf= rf(random_state=0)

crf.fit(X_train_perm,y_train_perm)
# Predicting the Test set results
y_pred = crf.predict(X_test_perm)

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_perm, y_pred)
print(cm)

#Recursive Feature Elimination with Cross Validations
from sklearn import datasets
from sklearn import metrics
from sklearn import linear_model
from sklearn.feature_selection import RFE
#K-fold cross validation with feature selection for each split
X_folds = np.array_split(X_train_perm, 3)
y_folds = np.array_split(y_train_perm, 3)


scores=[]
subset_all=[]
for k in range(3):
     X_train = list(X_folds)
     X_test  = X_train.pop(k)
     X_train = np.concatenate(X_train)
     y_train = list(y_folds)
     y_test  = y_train.pop(k)
     y_train = np.concatenate(y_train)
     rfe=RFE(classifier,11)
     rfe_fit=rfe.fit(X_train,y_train)
     subset=[]
     for i in range(0,len(rfe_fit.ranking_)):
         if rfe_fit.ranking_[i]==True:
             subset.append(i)
     subset_all.append(subset)
     X_feat = X_train[:,subset]
     X_feat_test=X_test[:,subset]
     scores.append(classifier.fit(X_feat, y_train).score(X_feat_test, y_test))
print(np.mean(scores))


# Compute ROC curve and ROC area for each class
from sklearn.metrics import roc_curve, auc


y_test_perm=np.array(y_test_perm)
y_pred=np.array(y_pred)
y_score = np.array(classifier.fit(X_train_perm, y_train_perm).decision_function(X_test_perm))

n_classes=y.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()

# Compute micro-average ROC curve and ROC area
for i in range(0,n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_perm[:, i], y_score[:])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_perm.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange',
         lw=lw, label='ROC curve (area = 0.72))')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Predicting Default Status')
plt.legend(loc="lower right")
plt.show()

