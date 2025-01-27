#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Load datasets
customers = pd.read_csv('Customers.csv')
products = pd.read_csv('Products.csv')
transactions = pd.read_csv('Transactions.csv')


# In[34]:


print("Customers Dataset:")
print(customers.info())
print(customers.head())


# In[6]:


print("\nProducts Dataset:")
print(products.info())
print(products.head())


# In[7]:


print("\nTransactions Dataset:")
print(transactions.info())
print(transactions.head())


# In[11]:


print("\nMissing Values:")
print('*'*40)
print("Customers:", customers.isnull().sum())
print('*'*40)
print("Products:", products.isnull().sum())
print('*'*40)
print("Transactions:", transactions.isnull().sum())


# In[13]:


# Merge Customers with Transactions
merged_data = transactions.merge(customers, on='CustomerID', how='left')

# Merge the resulting dataframe with Products
merged_data = merged_data.merge(products, on='ProductID', how='left')

# Display the merged data
print("\nMerged Dataset:")
print('*'*40)
print(merged_data.info())
print('*'*40)
print(merged_data.head())


# # Step 3: Perform EDA

# In[14]:


merged_data.head()


# In[18]:


merged_data.drop(columns='Price_x',inplace=True)


# In[40]:


merged_data.rename(columns={'Price_y':'Price'},inplace=True)
merged_data


# #### 1. Customer Demographics

# In[20]:


# Count of customers by region
region_counts = customers['Region'].value_counts()

# Visualization
region_counts.plot(kind='bar', title='Number of Customers by Region', color='skyblue')
plt.xlabel('Region')
plt.ylabel('Number of Customers')
plt.show()


# #### 2 Top-Selling Products

# In[45]:


# Top-selling products
top_products = merged_data.groupby('ProductName')['Quantity'].sum().sort_values(ascending=False).head(10)

# Visualization
top_products.plot(kind='bar', title='Top-Selling Products', color='orange')
plt.xlabel('Product')
plt.ylabel('Total Quantity Sold')
plt.show()


# #### 3. Total Sales by Region

# In[22]:


# Total sales by region
sales_by_region = merged_data.groupby('Region')['TotalValue'].sum().sort_values(ascending=False)

# Visualization
sales_by_region.plot(kind='bar', title='Total Sales by Region', color='green')
plt.xlabel('Region')
plt.ylabel('Total Sales (USD)')
plt.show()


# #### 4. Sales Trends Over Time

# In[27]:


# Convert TransactionDate to datetime
merged_data['TransactionDate'] = pd.to_datetime(merged_data['TransactionDate'])

# Extract year and month
merged_data['YearMonth'] = merged_data['TransactionDate'].dt.month

# Monthly sales
monthly_sales = merged_data.groupby('YearMonth')['TotalValue'].sum()

# Visualization
monthly_sales.plot(kind='line', title='Monthly Sales Trends', marker='o', color='blue')
plt.xlabel('Month')
plt.ylabel('Total Sales (USD)')
plt.show()


# #### 5. Category Contribution

# In[28]:


# Revenue by category
category_revenue = merged_data.groupby('Category')['TotalValue'].sum()

# Visualization
category_revenue.plot(kind='pie', title='Revenue by Product Category', autopct='%1.1f%%', startangle=90, colors=['gold', 'lightblue', 'lightgreen'])
plt.ylabel('')
plt.show()


# #### Customer Lifetime Value (CLV)

# In[29]:


# Calculate Customer Lifetime Value (CLV)
customer_lifetime_value = merged_data.groupby('CustomerID')['TotalValue'].sum().sort_values(ascending=False)

# Top 10 high-value customers
top_customers = customer_lifetime_value.head(10)

# Visualization
top_customers.plot(kind='bar', title='Top 10 High-Value Customers', color='purple')
plt.xlabel('CustomerID')
plt.ylabel('Total Spending (USD)')
plt.show()


# #### Repeat Customers vs. One-Time Buyers

# In[30]:


# Count transactions per customer
customer_transactions = merged_data.groupby('CustomerID')['TransactionID'].count()

# Categorize customers
repeat_customers = customer_transactions[customer_transactions > 1].count()
one_time_customers = customer_transactions[customer_transactions == 1].count()

# Visualization
plt.pie([repeat_customers, one_time_customers], labels=['Repeat Customers', 'One-Time Buyers'], autopct='%1.1f%%', startangle=90, colors=['lightblue', 'pink'])
plt.title('Repeat Customers vs. One-Time Buyers')
plt.show()


# #### Revenue Per Product Category Over Time

# In[31]:


# Revenue by category and month
category_monthly_revenue = merged_data.groupby(['YearMonth', 'Category'])['TotalValue'].sum().unstack()

# Visualization
category_monthly_revenue.plot(kind='line', marker='o', figsize=(10, 6))
plt.title('Revenue Trends by Category Over Time')
plt.xlabel('Month')
plt.ylabel('Revenue (USD)')
plt.legend(title='Category')
plt.show()


# #### Product Profitability

# In[43]:


# Average revenue per unit
product_profitability = merged_data.groupby('ProductName').agg({'TotalValue': 'sum', 'Quantity': 'sum'})
product_profitability['RevenuePerUnit'] = product_profitability['TotalValue'] / product_profitability['Quantity']

# Top 10 profitable products
top_profitable_products = product_profitability.sort_values('RevenuePerUnit', ascending=False).head(10)

# Visualization
top_profitable_products['RevenuePerUnit'].plot(kind='bar', color='darkgreen', title='Top 10 Profitable Products')
plt.xlabel('Product')
plt.ylabel('Revenue Per Unit (USD)')
plt.show()


# #### Peak Purchasing Days

# In[36]:


# Extract day of the week
merged_data['DayOfWeek'] = merged_data['TransactionDate'].dt.day_name()

# Count transactions by day
transactions_by_day = merged_data['DayOfWeek'].value_counts()

# Visualization
transactions_by_day.plot(kind='bar', color='teal', title='Transactions by Day of the Week')
plt.xlabel('Day')
plt.ylabel('Number of Transactions')
plt.show()


# #### Price Sensitivity Analysis

# In[46]:


# Calculate price variation
price_variation = merged_data.groupby('ProductName')['Price'].std().sort_values(ascending=False)

# Top 10 products with price variations
top_price_variations = price_variation.head(10)

# Visualization
top_price_variations.plot(kind='bar', color='maroon', title='Top 10 Products with Price Variations')
plt.xlabel('Product')
plt.ylabel('Price Standard Deviation (USD)')
plt.show()


# In[ ]:




