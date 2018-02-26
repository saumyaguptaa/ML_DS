
# coding: utf-8

# In[3]:


messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]
print(len(messages))


# In[4]:


for message_no, message in enumerate(messages[:10]):
    print(message_no, message)
    print('\n')


# In[6]:


import pandas as pd


# In[7]:


messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',
                           names=["label", "message"])
messages.head()


# In[8]:


messages.describe()


# In[9]:


messages.groupby('label').describe()


# In[10]:


messages['length'] = messages['message'].apply(len)
messages.head()


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')


# In[12]:


messages['length'].plot(bins=50, kind='hist') 


# In[13]:


messages.length.describe()


# Woah! 910 characters, let's use masking to find this message:

# In[14]:


messages[messages['length'] == 910]['message'].iloc[0]


# Looks like we have some sort of Romeo sending texts! But let's focus back on the idea of trying to see if message length is a distinguishing feature between ham and spam:

# In[18]:


messages.hist(column='length', by='label', bins=50,figsize=(12,4))


# In[19]:


import string

mess = 'Sample message! Notice: it has punctuation.'

nopunc = [char for char in mess if char not in string.punctuation]

nopunc = ''.join(nopunc)


# In[20]:


from nltk.corpus import stopwords
stopwords.words('english')[0:10] 


# In[21]:


nopunc.split()


# In[22]:



clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[23]:


clean_mess


# In[24]:


def text_process(mess):
    
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[25]:


messages.head()


# In[27]:



messages.head()


# In[28]:


from sklearn.feature_extraction.text import CountVectorizer


# In[31]:


# Might take awhile...
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])

# Print total number of vocab words
print(len(bow_transformer.vocabulary_))


# In[32]:


message4 = messages['message'][3]
print(message4)


# In[34]:


bow4 = bow_transformer.transform([message4])
print(bow4)
print(bow4.shape)


# In[36]:


print(bow_transformer.get_feature_names()[4073])
print(bow_transformer.get_feature_names()[9570])


# In[39]:


messages_bow = bow_transformer.transform(messages['message'])


# In[40]:


print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)


# In[46]:


sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(round(sparsity)))


# In[48]:


from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)


# In[50]:


print(tfidf_transformer.idf_[bow_transformer.vocabulary_['u']])
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])


# To transform the entire bag-of-words corpus into TF-IDF corpus at once:

# In[51]:


messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)


# In[52]:


from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])


# In[54]:


print('predicted:', spam_detect_model.predict(tfidf4)[0])
print('expected:', messages.label[3])


# In[55]:


all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)


# In[56]:


from sklearn.metrics import classification_report
print (classification_report(messages['label'], all_predictions))


# In[57]:


from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2)

print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))


# In[58]:


from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


# In[59]:


pipeline.fit(msg_train,label_train)


# In[60]:


predictions = pipeline.predict(msg_test)


# In[61]:


print(classification_report(predictions,label_test))

