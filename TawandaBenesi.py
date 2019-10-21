#!/usr/bin/env python
# coding: utf-8

# In[273]:


# Import useful packages

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import random


# In[274]:


# Load the data set
dat1 = pd.read_csv("C:/Users/tbene/Downloads/ShelfLifeData.csv")

# View the column names 
print(dat1.columns)

# Create a binary target variable with 1 (No difference from fresh) and 0 otherwise
dat1['target'] = np.array(list(map(int,dat1['Difference From Fresh'] < 20)))

# Drop some ID variables, numeric target variable and other not so useful predictors
dat1 = dat1.drop(['Study Number', 'Sample ID','Prediction','Transparent Window in Package','Difference From Fresh',
                 'Prediction'], axis=1)
# Print DataFrame information

dat1_info = dat1.info()
print(dat1_info)

print("\n")

# Inspect missing values in the dataset and print summary statistics

dat1_summary = dat1.describe()
print(dat1_summary)

print("\n")
print(dat1.head())

# Create tables and count missing values for categorical features
# Also seperate the categorical and quantitative features for further preprocessing

categorical_features = pd.DataFrame()
quant_features = pd.DataFrame()
for col in dat1:
    # Compare if the dtype is object
    if dat1[col].dtype=='object' or dat1[col].dtype=='int32':
        categorical_features[col]=dat1[col]
        print(col) 
        print('Missing count:'+ str(dat1[col].isna().sum()))
        print(dat1[col].value_counts())
    else:
        quant_features[col] = dat1[col]
            
# Separate the target variable into variable Y
Y = categorical_features['target']  
#Drop target variable from categorical features
categorical_features=categorical_features.drop(['target'], axis=1) 


# In[275]:


# Observe the distribution of categorical variables grouped by target variable
for col in categorical_features:
    # Compare if the dtype is object
    print(dat1.groupby(['target',col]).size())


# In[277]:


# Visualize the quantitative variable colored by target variable
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.pairplot(dat1, hue='target')
print('Some of the bad features will be fixed by Scaling')


# In[278]:


# Impute the missing values with meadian imputation and most_freq imputation
# Also recode string type to numeric type

from sklearn_pandas import CategoricalImputer
from sklearn.impute import SimpleImputer

le = LabelEncoder()
cat_imputer = CategoricalImputer()
quant_features.fillna(quant_features.median(), inplace=True)

for col in categorical_features:
    categorical_features[col]=cat_imputer.fit_transform(categorical_features[col])
    categorical_features[col]=le.fit_transform(categorical_features[col])
    
#one_hot_dat = pd.DataFrame(onehot_encoder.fit_transform(categorical_features))  
#categorical_features = pd.get_dummies(categorical_features, drop_first=True)

dat1 = pd.concat([quant_features, categorical_features], axis=1)


# In[279]:


# Preprocess the Features by Scaling using MinMaxScaler
X = dat1

# Instantiate MinMaxScaler and use it to rescale
scaler = MinMaxScaler(copy=True,feature_range=(0,1))
rescaledX = scaler.fit_transform(X)
X = rescaledX


# In[280]:


# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=2020)


# In[281]:


# Train a variety of base models and choose the best on test data
from sklearn import model_selection
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix

seed = 2020
dt = DecisionTreeClassifier()
# prepare models
models = []
models.append(('LR  ', LogisticRegression(solver='liblinear', random_state=seed)))
models.append(('LDA ', LinearDiscriminantAnalysis()))
models.append(('KNN ', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=seed)))
models.append(('NB  ', GaussianNB()))
models.append(('SVM ', SVC(gamma='auto',random_state=seed)))
models.append(('RFC ', RandomForestClassifier(n_estimators=500,random_state=seed)))
models.append(('GBC ', GradientBoostingClassifier(random_state=seed)))
models.append(('BGC ', BaggingClassifier(base_estimator=dt,oob_score=False,n_estimators=500,n_jobs=-1,random_state=seed)))
models.append(('ADAB', AdaBoostClassifier(base_estimator=dt,random_state=seed,n_estimators=500)))
models.append(('XGBC', XGBClassifier(random_state=seed)))

# evaluate each model in turn

names = []
for name, model in models:
    model_fit = model.fit(X_train,y_train)
    model_predict = model.predict(X_test)
    # accuracy_score
    score_1 = accuracy_score(y_test, model_predict)
    #score_accuracy.append(score_1)
    # f1 score
    score_2 = f1_score(y_test, model_predict)
    #score_f1.append(score_2)
    # recall_score
    score_3 = recall_score(y_test, model_predict)
    #score_recall.append(score_3)
    # precision_score
    score_4 = precision_score(y_test, model_predict)
    #score_precision.append(score_3)
    names.append(name)
    print(confusion_matrix(y_test,model_predict))
    print(name, 'Accuracy: ', round(score_1,4),' F1 score:  ', round(score_2,4),
         ' Precision:  ', round(score_4,4),' Recall:  ', round(score_3,4))

vc = VotingClassifier(estimators=models)
# Fit 'vc' to the traing set
vc.fit(X_train, y_train)
# Predict test set labels
y_pred = vc.predict(X_test)
# Evaluate the test-set accuracy of 'vc'
# accuracy_score
score_1 = accuracy_score(y_test, y_pred)
# f1 score
score_2 = f1_score(y_test, y_pred)
# recall_score
score_3 = recall_score(y_test, y_pred)
# precision_score
score_4 = precision_score(y_test, y_pred)
print(confusion_matrix(y_test,y_pred))
print('VC  ', 'Accuracy: ', round(score_1,4),' F1 score:  ', round(score_2,4),
         ' Precision:  ', round(score_4,4),' Recall:  ', round(score_3,4))

print('The data is skewed and further preprocessing is necessary')


# In[282]:


# The target variable is imbalanced. Time to reshape the data by Upsampling the minority

from sklearn.utils import resample
#Y = dat1['target']

#X = dat1.drop(['target'], axis=1)
X=dat1
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=2020)
# concatenate our training data back together
train_dat = pd.concat([X_train, y_train], axis=1)
# separate minority and majority classes
notfresh = train_dat[train_dat.target==0]
fresh = train_dat[train_dat.target==1]

# upsample minority
notfresh_upsampled = resample(notfresh,
                          replace=True, # sample with replacement
                          n_samples=len(fresh), # match number in majority class
                          random_state=2020) # reproducible results

# combine majority and upsampled minority
dat1_upsampled = pd.concat([fresh, notfresh_upsampled])

# check new class counts
print(dat1_upsampled.target.value_counts())

y_train = dat1_upsampled.target
X_train = dat1_upsampled.drop('target', axis=1)


# In[283]:


seed = 2020
# prepare models
models = []
models.append(('LR  ', LogisticRegression(solver='liblinear', random_state=seed)))
models.append(('LDA ', LinearDiscriminantAnalysis()))
models.append(('KNN ', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=seed)))
models.append(('NB  ', GaussianNB()))
models.append(('SVM ', SVC(gamma='auto',random_state=seed)))
models.append(('RFC ', RandomForestClassifier(n_estimators=500,random_state=seed)))
models.append(('GBC ', GradientBoostingClassifier(n_estimators=500,max_depth=5,random_state=seed)))
models.append(('BGC ', BaggingClassifier(base_estimator=dt, n_estimators=500,oob_score=True,n_jobs=-1,random_state=seed)))
models.append(('ADAB',AdaBoostClassifier(base_estimator=dt, n_estimators=500,random_state=seed)))
models.append(('XGBC', XGBClassifier(random_state=seed)))
# evaluate each model in turn

names = []
for name, model in models:
    model_fit = model.fit(X_train,y_train)
    model_predict = model.predict(X_test)
    # accuracy_score
    score_1 = accuracy_score(y_test, model_predict)
    #score_accuracy.append(score_1)
    # f1 score
    score_2 = f1_score(y_test, model_predict)
    #score_f1.append(score_2)
    # recall_score
    score_3 = recall_score(y_test, model_predict)
    #score_recall.append(score_3)
    # precision_score
    score_4 = precision_score(y_test, model_predict)
    #score_precision.append(score_3)
    names.append(name)
    print(confusion_matrix(y_test,model_predict))
    print(name, 'Accuracy: ', round(score_1,4),' F1 score:  ', round(score_2,4),
         ' Precision:  ', round(score_4,4),' Recall:  ', round(score_3,4))

vc = VotingClassifier(estimators=models)
# Fit 'vc' to the traing set
vc.fit(X_train, y_train)
# Predict test set labels
y_pred = vc.predict(X_test)
# Evaluate the test-set accuracy of 'vc'
# accuracy_score
score_1 = accuracy_score(y_test, y_pred)
# f1 score
score_2 = f1_score(y_test, y_pred)
# recall_score
score_3 = recall_score(y_test, y_pred)
# precision_score
score_4 = precision_score(y_test, y_pred)
print(confusion_matrix(y_test,y_pred))
print('VC  ', 'Accuracy: ', round(score_1,4),' F1 score:  ', round(score_2,4),
         ' Precision:  ', round(score_4,4),' Recall:  ', round(score_3,4))
print('Quite resonable compared to previous data')


# In[285]:


# THIS SECTION IS THE SUMMARY OF THE FINAL MODEL BASED ON UPSAMPLING THE MINORITY CLASS 

print('The Voting Classifier (VC) is the most resonable compared to other models')
# Predict test set labels
y_pred = vc.predict(X)
# Evaluate the test-set accuracy of 'vc'
# accuracy_score
score_1 = accuracy_score(Y, y_pred)
# f1 score
score_2 = f1_score(Y, y_pred)
# recall_score
score_3 = recall_score(Y, y_pred)
# precision_score
score_4 = precision_score(Y, y_pred)
print(confusion_matrix(Y,y_pred))
print('VC  ', 'Accuracy: ', round(score_1,4),' F1 score:  ', round(score_2,4),
         ' Precision:  ', round(score_4,4),' Recall:  ', round(score_3,4))

dat2 = pd.read_csv("C:/Users/tbene/Downloads/ShelfLifeData.csv")
dat2['Prediction'] = y_pred

dat2.head()
dat2.to_csv(r'TawandaBenesiPrediction.csv')


# In[286]:


# Let's try Downsampling the majority

# downsample majority
fresh_downsampled = resample(fresh,
                                replace = False, # sample without replacement
                                n_samples = len(notfresh), # match minority n
                                random_state = 2020) # reproducible results

# combine minority and downsampled majority
dat1_downsampled = pd.concat([fresh_downsampled, notfresh])

# checking counts
print(dat1_downsampled.target.value_counts())
y_train = dat1_downsampled.target
X_train = dat1_downsampled.drop('target', axis=1)


# In[287]:


seed = 2020
# prepare models
models = []
models.append(('LR  ', LogisticRegression(solver='liblinear', random_state=seed)))
models.append(('LDA ', LinearDiscriminantAnalysis()))
models.append(('KNN ', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=seed)))
models.append(('NB  ', GaussianNB()))
models.append(('SVM ', SVC(gamma='auto',random_state=seed)))
models.append(('RFC ', RandomForestClassifier(n_estimators=500,random_state=seed)))
models.append(('GBC ', GradientBoostingClassifier(n_estimators=500,max_depth=5,random_state=seed)))
models.append(('BGC ', BaggingClassifier(base_estimator=dt, n_estimators=500,oob_score=True,n_jobs=-1,random_state=seed)))
models.append(('ADAB',AdaBoostClassifier(base_estimator=dt, n_estimators=500,random_state=seed)))
models.append(('XGBC', XGBClassifier(random_state=seed)))
# evaluate each model in turn

names = []
for name, model in models:
    model_fit = model.fit(X_train,y_train)
    model_predict = model.predict(X_test)
    # accuracy_score
    score_1 = accuracy_score(y_test, model_predict)
    #score_accuracy.append(score_1)
    # f1 score
    score_2 = f1_score(y_test, model_predict)
    #score_f1.append(score_2)
    # recall_score
    score_3 = recall_score(y_test, model_predict)
    #score_recall.append(score_3)
    # precision_score
    score_4 = precision_score(y_test, model_predict)
    #score_precision.append(score_3)
    names.append(name)
    print(confusion_matrix(y_test,model_predict))
    print(name, 'Accuracy: ', round(score_1,4),' F1 score:  ', round(score_2,4),
         ' Precision:  ', round(score_4,4),' Recall:  ', round(score_3,4))

vc = VotingClassifier(estimators=models)
# Fit 'vc' to the traing set
vc.fit(X_train, y_train)
# Predict test set labels
y_pred = vc.predict(X_test)
# Evaluate the test-set accuracy of 'vc'
# accuracy_score
score_1 = accuracy_score(y_test, y_pred)
# f1 score
score_2 = f1_score(y_test, y_pred)
# recall_score
score_3 = recall_score(y_test, y_pred)
# precision_score
score_4 = precision_score(y_test, y_pred)
print(confusion_matrix(y_test,y_pred))
print('VC  ', 'Accuracy: ', round(score_1,4),' F1 score:  ', round(score_2,4),
         ' Precision:  ', round(score_4,4),' Recall:  ', round(score_3,4))

print('Not as good as upsampling the minority')


# In[288]:


# Let's try SMOTE technique
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2020, ratio=1.0)
X = dat1
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=2020)
X_train, y_train = sm.fit_sample(X_train, y_train)


# In[289]:


seed = 2020
# prepare models
models = []
models.append(('LR  ', LogisticRegression(solver='liblinear', random_state=seed)))
models.append(('LDA ', LinearDiscriminantAnalysis()))
models.append(('KNN ', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=seed)))
models.append(('NB  ', GaussianNB()))
models.append(('SVM ', SVC(gamma='auto',random_state=seed)))
models.append(('RFC ', RandomForestClassifier(n_estimators=500,random_state=seed)))
models.append(('GBC ', GradientBoostingClassifier(n_estimators=500,max_depth=5,random_state=seed)))
models.append(('BGC ', BaggingClassifier(base_estimator=dt, n_estimators=500,oob_score=True,n_jobs=-1,random_state=seed)))
models.append(('ADAB',AdaBoostClassifier(base_estimator=dt, n_estimators=500,random_state=seed)))
#models.append(('XGBC', XGBClassifier(random_state=seed)))
# evaluate each model in turn

names = []
for name, model in models:
    model_fit = model.fit(X_train,y_train)
    model_predict = model.predict(X_test)
    # accuracy_score
    score_1 = accuracy_score(y_test, model_predict)
    #score_accuracy.append(score_1)
    # f1 score
    score_2 = f1_score(y_test, model_predict)
    #score_f1.append(score_2)
    # recall_score
    score_3 = recall_score(y_test, model_predict)
    #score_recall.append(score_3)
    # precision_score
    score_4 = precision_score(y_test, model_predict)
    #score_precision.append(score_3)
    names.append(name)
    print(confusion_matrix(y_test,model_predict))
    print(name, 'Accuracy: ', round(score_1,4),' F1 score:  ', round(score_2,4),
         ' Precision:  ', round(score_4,4),' Recall:  ', round(score_3,4))

vc = VotingClassifier(estimators=models)
# Fit 'vc' to the traing set
vc.fit(X_train, y_train)
# Predict test set labels
y_pred = vc.predict(X_test)
# Evaluate the test-set accuracy of 'vc'
# accuracy_score
score_1 = accuracy_score(y_test, y_pred)
# f1 score
score_2 = f1_score(y_test, y_pred)
# recall_score
score_3 = recall_score(y_test, y_pred)
# precision_score
score_4 = precision_score(y_test, y_pred)
print(confusion_matrix(y_test,y_pred))
print('VC  ', 'Accuracy: ', round(score_1,4),' F1 score:  ', round(score_2,4),
         ' Precision:  ', round(score_4,4),' Recall:  ', round(score_3,4))
print('Not as good as upsampling the minority')

