#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


# In[12]:


# Load datasets
customers = pd.read_csv("Customers.csv")
products = pd.read_csv("Products.csv")
transactions = pd.read_csv("Transactions.csv")


# In[13]:


# Merge datasets
merged = transactions.merge(customers, on="CustomerID").merge(products, on="ProductID")


# In[14]:


# Feature engineering
customer_features = merged.groupby("CustomerID").agg(
    TotalSpent=("TotalValue", "sum"),
    AvgQuantity=("Quantity", "mean"),
    MostFrequentCategory=("Category", lambda x: x.mode()[0])
).reset_index()


# In[15]:


customer_features


# In[16]:


# One-hot encode categorical features
encoder = OneHotEncoder()
encoded_region = pd.DataFrame(encoder.fit_transform(customers[["Region"]]).toarray(), columns=encoder.get_feature_names_out())
encoded_category = pd.DataFrame(encoder.fit_transform(customer_features[["MostFrequentCategory"]]).toarray(), columns=encoder.get_feature_names_out())


# In[17]:


encoded_category


# In[18]:


# Combine features
features = pd.concat([
    customer_features.drop(columns=["MostFrequentCategory"]),
    encoded_region,
    encoded_category
], axis=1)


# In[19]:


# Normalize numerical features
scaler = MinMaxScaler()
features[["TotalSpent", "AvgQuantity"]] = scaler.fit_transform(features[["TotalSpent", "AvgQuantity"]])


# In[24]:


features.dropna(inplace=True)

print(features.isnull().sum())


# In[25]:


# Compute similarity matrix
similarity_matrix = cosine_similarity(features.drop(columns=["CustomerID"]))
similarity_df = pd.DataFrame(similarity_matrix, index=features["CustomerID"], columns=features["CustomerID"])


# In[26]:


# Generate lookalike recommendations
lookalike_results = {}
for customer_id in features["CustomerID"].head(20):  # First 20 customers
    similar_customers = similarity_df[customer_id].sort_values(ascending=False).iloc[1:4]
    lookalike_results[customer_id] = [(index, score) for index, score in similar_customers.items()]


# In[27]:


# Save to CSV
lookalike_df = pd.DataFrame([
    {"CustomerID": cust, "Lookalikes": str(lookalike_results[cust])}
    for cust in lookalike_results
])
lookalike_df.to_csv("Lookalike.csv", index=False)

print("Lookalike.csv created successfully.")


# In[ ]:




