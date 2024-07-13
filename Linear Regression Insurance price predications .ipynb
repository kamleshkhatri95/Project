#!/usr/bin/env python
# coding: utf-8

# Importing Libraries

# In[209]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[171]:


df=pd.read_csv(r"C:\Users\DSCKK2\OneDrive - CELANESE CORPORATION\Python\EXCEL\new_insurance_data.csv")


# In[172]:


df.head()


# In[173]:


df.shape


# In[174]:


df.info()


# In[175]:


df.isnull().sum()


# In[176]:


round(df.isnull().sum()/len(df)*100,2)


# removing Null values from data

# In[177]:


df.dropna(inplace=True)


# In[178]:


df.isnull().sum()


# sns.boxplot(data=df,x='age')

# In[179]:


for i in df.columns:
    if df[i].dtype!='object':
        sns.boxplot(data=df,x=i)
        plt.show()
 


# In[180]:


df.columns


# In[181]:


out_list=['bmi', 'past_consultations','Hospital_expenditure', 'Anual_Salary',  'charges']


# In[182]:


for col in out_list:
    Q1=df[col].quantile(0.25)
    Q3=df[col].quantile(0.75)
    
    IQR=Q3-Q1
    
    UL=Q3+1.5*IQR
    LL=Q1-1.5*IQR
    
    df=df[(df[col]>LL)&(df[col]<UL)]


# In[183]:


for col in out_list:
    sns.boxplot(data=df,x=col)
    plt.show()


# In[184]:


df.corr(numeric_only=True)


# In[185]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='Blues')


# In[186]:


df.drop('num_of_steps',axis=1,inplace=True)


# In[187]:


sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='Blues')


# Label encoding

# In[188]:


df['sex']=df['sex'].map({'male':0,'female':1})


# In[191]:


df['region'].unique()


# In[190]:


df['smoker']=df['smoker'].map({'yes':0,'no':1})
df['region']=df['region'].map({'southeast':0,'southwest':1,'northwest':2,'northeast':3})


# Model building

# In[192]:


x=df.drop('charges',axis=1)


# In[193]:


y=df['charges']


# In[194]:


from sklearn.model_selection import train_test_split


# In[195]:


x_train,x_test,y_train,y_test=train_test_split(x,y ,train_size=0.75,random_state=100)


# In[196]:


from sklearn.linear_model import LinearRegression


# In[197]:


Le=LinearRegression()


# In[198]:


Le.fit(x_train,y_train)


# In[201]:


pred=Le.predict(x_test)


# In[202]:


pred


# In[203]:


from sklearn.metrics import r2_score
r2_score(y_test,pred)


# In[ ]:




