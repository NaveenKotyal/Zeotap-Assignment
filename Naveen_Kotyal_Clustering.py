#!/usr/bin/env python
# coding: utf-8

# ##  Data Preparation

# In[35]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# In[36]:


# Load data
customers = pd.read_csv('Customers.csv')
transactions = pd.read_csv('Transactions.csv')


# In[37]:


# Merge datasets
data = pd.merge(customers, transactions, on='CustomerID')
data


# ## Feature Engineering

# In[38]:


from datetime import datetime

# Recency
data['TransactionDate'] = pd.to_datetime(data['TransactionDate'])
recency = data.groupby('CustomerID')['TransactionDate'].max()
recency = (datetime.now() - recency).dt.days.rename('Recency')

# Frequency
frequency = data.groupby('CustomerID').size().rename('Frequency')

# Tenure
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
tenure = (datetime.now() - customers.set_index('CustomerID')['SignupDate']).dt.days.rename('Tenure')

# Combine features
features = pd.concat([recency, frequency, tenure], axis=1).reset_index()


# In[39]:


features.isnull().sum()


# In[40]:


features.dropna(inplace=True)


# ## Clustering

# In[41]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features[['Recency', 'Frequency', 'Tenure']])


# In[55]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
features['Cluster'] = kmeans.fit_predict(features_scaled)


# In[43]:


features


# ## Evaluation Metrics

# In[44]:


from sklearn.metrics import davies_bouldin_score
db_index = davies_bouldin_score(features_scaled, features['Cluster'])
print(f"Davies-Bouldin Index: {db_index}")


# In[45]:


from sklearn.metrics import silhouette_score
silhouette = silhouette_score(features_scaled, features['Cluster'])
print(f"Silhouette Score: {silhouette}")


# In[46]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_scaled)
features['PCA1'], features['PCA2'] = pca_result[:, 0], pca_result[:, 1]

sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=features, palette='viridis')
plt.title('Customer Segmentation Clusters')
plt.show()


# In[47]:


features['Cluster'].value_counts().plot(kind='bar')
plt.title('Cluster Distribution')
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.show()


# In[ ]:




