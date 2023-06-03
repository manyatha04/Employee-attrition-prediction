#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

import missingno as msno

# configure
# sets matplotlib to inline and displays graphs below the corresponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

# import the necessary modelling algos.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV

from imblearn.over_sampling import SMOTE

# preprocess
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder

# ANN and DL libraries
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop
from keras.utils import to_categorical

import tensorflow as tf
import random as rn


# In[3]:


df=pd.read_csv('Employee-Attrition.csv')


# In[4]:


df.head()


# In[ ]:


df.info()  # no null or Nan values.


# In[ ]:


df.isnull().sum()


# In[ ]:


msno.matrix(df) # just to visualize.


# In[ ]:


sns.factorplot(data=df,kind='box',size=10,aspect=3)


# In[ ]:


sns.distplot(df['Age'])


# In[ ]:


warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

fig,ax = plt.subplots(5,2, figsize=(9,9))                
sns.distplot(df['TotalWorkingYears'], ax = ax[0,0]) 
sns.distplot(df['MonthlyIncome'], ax = ax[0,1]) 
sns.distplot(df['YearsAtCompany'], ax = ax[1,0]) 
sns.distplot(df['DistanceFromHome'], ax = ax[1,1]) 
sns.distplot(df['YearsInCurrentRole'], ax = ax[2,0]) 
sns.distplot(df['YearsWithCurrManager'], ax = ax[2,1]) 
sns.distplot(df['YearsSinceLastPromotion'], ax = ax[3,0]) 
sns.distplot(df['PercentSalaryHike'], ax = ax[3,1]) 
sns.distplot(df['YearsSinceLastPromotion'], ax = ax[4,0]) 
sns.distplot(df['TrainingTimesLastYear'], ax = ax[4,1]) 
plt.tight_layout()
plt.show()


# In[ ]:


cat_df=df.select_dtypes(include='object')


# In[ ]:


cat_df.columns


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)


# In[ ]:


def plot_cat(attr,labels=None):
    if(attr=='JobRole'):
        sns.factorplot(data=df,kind='count',size=5,aspect=3,x=attr)
        return
    
    sns.factorplot(data=df,kind='count',size=5,aspect=1.5,x=attr)


# In[ ]:


plot_cat('Attrition')   


# In[ ]:


plot_cat('BusinessTravel')   


# In[ ]:


#corelation matrix.
cor_mat= df.corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)


# In[ ]:


df.columns


# In[ ]:


sns.factorplot(data=df,y='Age',x='Attrition',size=5,aspect=1,kind='box')


# In[ ]:


df.Department.value_counts()


# In[ ]:


sns.factorplot(data=df,kind='count',x='Attrition',col='Department')


# In[ ]:


pd.crosstab(columns=[df.Attrition],index=[df.Department],margins=True,normalize='index') # set normalize=index to view rowwise %.


# In[ ]:


pd.crosstab(columns=[df.Attrition],index=[df.Gender],margins=True,normalize='index') # set normalize=index to view rowwise %.


# In[ ]:


pd.crosstab(columns=[df.Attrition],index=[df.JobLevel],margins=True,normalize='index') # set normalize=index to view rowwise %.


# In[ ]:


sns.factorplot(data=df,kind='bar',x='Attrition',y='MonthlyIncome')


# #### 3.1.6 ) Job Satisfaction

# In[ ]:


sns.factorplot(data=df,kind='count',x='Attrition',col='JobSatisfaction')


# In[ ]:


pd.crosstab(columns=[df.Attrition],index=[df.JobSatisfaction],margins=True,normalize='index') # set normalize=index to view rowwise %.


# #### 3.1.7 ) Environment Satisfaction 

# In[ ]:


pd.crosstab(columns=[df.Attrition],index=[df.EnvironmentSatisfaction],margins=True,normalize='index') # set normalize=index to view rowwise %.


# Again we can notice that the relative percent of 'No' in people with higher grade of environment satisfacftion.

# #### 3.1.8 ) Job Involvement

# In[ ]:


pd.crosstab(columns=[df.Attrition],index=[df.JobInvolvement],margins=True,normalize='index') # set normalize=index to view rowwise %.


# #### 3.1.9 ) Work Life Balance

# In[ ]:


pd.crosstab(columns=[df.Attrition],index=[df.WorkLifeBalance],margins=True,normalize='index') # set normalize=index to view rowwise %.


# In[ ]:


pd.crosstab(columns=[df.Attrition],index=[df.RelationshipSatisfaction],margins=True,normalize='index') # set normalize=index to view rowwise %.


# In[1]:


# Splitting the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(df.drop('Attrition', axis=1), df['Attrition'].values, test_size=0.25, random_state=42)

# Encoding categorical features using LabelEncoder
for col in cat_df.columns:
    transform(col)

# Scaling the numerical features using StandardScaler
scaler = StandardScaler()
scaled_x_train = scaler.fit_transform(x_train.select_dtypes(include=np.number))
scaled_x_test = scaler.transform(x_test.select_dtypes(include=np.number))

# Concatenating the scaled numerical features and encoded categorical features
x_train = np.concatenate([scaled_x_train, x_train.select_dtypes(include='object').apply(LabelEncoder().fit_transform)], axis=1)
x_test = np.concatenate([scaled_x_test, x_test.select_dtypes(include='object').apply(LabelEncoder().fit_transform)], axis=1)

# Printing original employee numbers of x_test
print("Original Employee Numbers of X_test:")
print(x_test[:, 0])


# In[ ]:


df.drop(['BusinessTravel','DailyRate','EmployeeCount','EmployeeNumber','HourlyRate','MonthlyRate'
           ,'NumCompaniesWorked','Over18','StandardHours', 'StockOptionLevel','TrainingTimesLastYear'],axis=1,inplace=True)


# In[ ]:


def transform(feature):
    le=LabelEncoder()
    df[feature]=le.fit_transform(df[feature])
    print(le.classes_)
    


# In[ ]:


cat_df=df.select_dtypes(include='object')
cat_df.columns


# In[ ]:


for col in cat_df.columns:
    transform(col)


# In[ ]:


df.head() # just to verify.


# In[ ]:


scaler = StandardScaler()
scaled_df = scaler.fit_transform(df.drop('Attrition', axis=1))
X = scaled_df
Y = df['Attrition'].values


# In[ ]:


Y=to_categorical(Y)
Y


# In[ ]:


np.random.seed(42)


# In[ ]:


rn.seed(42)


# In[ ]:


tf.random.set_seed(42)


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)


# In[ ]:


model=Sequential()
model.add(Dense(input_dim=23,units=8,activation='relu'))
model.add(Dense(units=16,activation='relu'))
model.add(Dense(units=2,activation='sigmoid'))


# In[ ]:


model.compile(optimizer=Adam(lr=0.01),loss='binary_crossentropy',metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


History=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,verbose=1)


# In[ ]:


y_pred = model.predict(x_test)
y_pred_binary = np.where(y_pred > 0.5, 1, 0)

print("Binary prediction for employee attrition:\n Stay | Leave")
print(y_pred_binary)


# In[ ]:


model.predict(x_test)


# In[ ]:


model.evaluate(x_test,y_test)


# In[ ]:


plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()


# In[ ]:


plt.plot(History.history['accuracy'])
plt.plot(History.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()


# In[ ]:


print(x_test[1])


# In[ ]:


# assume the trained model is already loaded as 'model'
import numpy as np
from sklearn.preprocessing import StandardScaler

# example data
data = np.array([[0.26, 0.7, 3., 238., 6., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1.]])

# standardize the data using the same scaler used during training
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)

# make a prediction
prediction = model.predict(data_standardized)

print(prediction)

