
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd


# In[25]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic('matplotlib inline')


# In[26]:


df = pd.read_csv('911.csv')


# In[27]:


df.info()


# In[28]:


df.head(3)


# In[29]:


df['zip'].value_counts().head(5)


# In[30]:


df['twp'].value_counts().head(5)


# In[31]:


df['title'].nunique()


# In[32]:


df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])


# In[33]:


df['Reason'].value_counts()


# In[34]:


sns.countplot(x='Reason',data=df,palette='viridis')


# In[35]:


type(df['timeStamp'].iloc[0])


# In[36]:


df['timeStamp'] = pd.to_datetime(df['timeStamp'])


# In[37]:


df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)


# In[38]:


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}


# In[39]:


df['Day of Week'] = df['Day of Week'].map(dmap)


# In[40]:


sns.countplot(x='Day of Week',data=df,hue='Reason',palette='viridis')

# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[41]:


sns.countplot(x='Month',data=df,hue='Reason',palette='viridis')

# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[42]:


# It is missing some months. 9,10, and 11 are not there.


# In[43]:


byMonth = df.groupby('Month').count()
byMonth.head()


# In[44]:


# Could be any column
byMonth['twp'].plot()


# In[45]:


sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())


# In[46]:


df['Date']=df['timeStamp'].apply(lambda t: t.date())


# In[47]:


df.groupby('Date').count()['twp'].plot()
plt.tight_layout()


# In[48]:


df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()


# In[49]:


df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()
plt.title('Fire')
plt.tight_layout()


# In[50]:


df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()
plt.title('EMS')
plt.tight_layout()


# In[51]:


dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
dayHour.head()


# In[52]:


plt.figure(figsize=(12,6))
sns.heatmap(dayHour,cmap='viridis')


# In[53]:


sns.clustermap(dayHour,cmap='viridis')


# In[54]:


dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()
dayMonth.head()


# In[55]:


plt.figure(figsize=(12,6))
sns.heatmap(dayMonth,cmap='viridis')


# In[56]:


sns.clustermap(dayMonth,cmap='viridis')

