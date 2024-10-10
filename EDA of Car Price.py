#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# In[4]:


train = pd.read_csv("C:/Users/hp/Downloads/playground-series-s4e9/train.csv")


# # Data Cleaning and Preprocessing

# In[5]:


train.head()


# In[6]:


train.info()


# In[7]:


train.describe().T


# In[8]:


train.dtypes


# In[9]:


numerical_column_names = train.select_dtypes(include=['number']).columns
print("Numerical Column Names:", numerical_column_names.tolist())


# In[10]:


object_column_names = train.select_dtypes(include=['object']).columns
print("Object Column Names:", object_column_names.tolist())


# In[11]:


train.isnull().sum()


# In[12]:


train.isnull().sum()


# In[13]:


print("Number of Rows:",train.shape[0])


# In[14]:


print("Number of Columns:",train.shape[1])


# In[15]:


train.nunique()


# In[16]:


categorical_columns = ['brand', 'model', 'fuel_type', 'engine', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title']


# In[17]:


for col in categorical_columns:
    
    print(f"Category in {col} is : {train[col].unique()}")
    print("<--- --- --- --- --- --- --- --- --- --->")


# In[18]:


train.drop('id', axis = 1, inplace = True)


# # Handling Null value

# In[20]:


train['clean_title'] = train['clean_title'].fillna('No')
train['accident'] = train['accident'].fillna('No Data')


# In[21]:


mode_value = train['fuel_type'].mode()[0]
train['fuel_type'].fillna(mode_value, inplace=True)


# #Univariate Analysis 

# In[22]:


print("Average Milage of the car:",train['milage'].mean().round(2))


# In[23]:


print("Maximum Milage of the car:",train['milage'].max())


# In[24]:


print("Average Price of the car:",train['price'].mean().round(2))


# In[25]:


print("Maximum Price of the car:",train['price'].max())


# # Exploratory Data Analysis

# # 1.1 Categorical Feature

# In[26]:


brand_counts = train['brand'].value_counts().reset_index().head(10)
brand_counts.columns = ['brand', 'count']


# In[27]:


fig = px.bar(brand_counts, x='brand', y='count', 
             title='Top 10 Brand',
             labels={'brand': 'Brand', 'count': 'Count'},
             color='count',  
             color_continuous_scale='Viridis'  
)

fig.update_layout(
    xaxis_title='Brand',
    yaxis_title='Count',
    width=600,  
    height=600   
)

fig.show()


# In[28]:


model_counts = train['model'].value_counts().reset_index().head(10)
model_counts.columns = ['model', 'count']


# In[29]:


fig = px.bar(model_counts, x='model', y='count', 
             title='Top 10 Model',
             labels={'model': 'Model', 'count': 'Count'},
             color='count',  
             color_continuous_scale='Viridis'  
)

fig.update_layout(
    xaxis_title='Model',
    yaxis_title='Count',
    width=600,  
    height=600,
)

fig.show()


# In[30]:


fuel_counts = train['fuel_type'].value_counts().reset_index()
fuel_counts.columns = ['fuel_type', 'count']


# In[31]:


fig = px.bar(fuel_counts, x='fuel_type', y='count', 
             title='Distribution of Fuel Count',
             labels={'fuel_type': 'Fuel Type', 'count': 'Count'},
             color='count',  
             color_continuous_scale='Viridis'  
)

fig.update_layout(
    xaxis_title='Fuel Type',
    yaxis_title='Count',
    width=600,  
    height=500,
)

fig.show()


# In[32]:


transmission_counts = train['transmission'].value_counts().reset_index().head(10)
transmission_counts.columns = ['transmission', 'count']


# In[33]:


fig = px.bar(transmission_counts, x='transmission', y='count', 
             title='Distribution of Transmission Type',
             labels={'transmission': 'Transmission Type', 'count': 'Count'},
             color='count', 
             color_continuous_scale='Viridis'  
)

fig.update_layout(
    xaxis_title='Transmission Type',
    yaxis_title='Count',
    width=600,  
    height=600,
)

fig.show()


# In[34]:


int_col_counts = train['int_col'].value_counts().reset_index().head(10)
int_col_counts.columns = ['int_col', 'count']


# In[35]:


fig = px.bar(int_col_counts, x='int_col', y='count', 
             title='Distribution of Interior Color Category',
             labels={'int_col': 'Color', 'count': 'Count'},
             color='count',  
             color_continuous_scale='Viridis'  
)

fig.update_layout(
    xaxis_title='Color',
    yaxis_title='Count',
    width=600,  
    height=500,
)

fig.show()


# In[36]:


ext_col_counts = train['ext_col'].value_counts().reset_index().head(10)
ext_col_counts.columns = ['ext_col', 'count']


# In[37]:


fig = px.bar(ext_col_counts, x='ext_col', y='count', 
             title='Distribution of Exterior Color Category',
             labels={'ext_col': 'Color', 'count': 'Count'},
             color='count',  
             color_continuous_scale='Viridis'  
)

fig.update_layout(
    xaxis_title='Color',
    yaxis_title='Count',
    width=600,  
    height=500,
)

fig.show()


# In[38]:


accident_counts = train['accident'].value_counts().reset_index()
accident_counts.columns = ['accident', 'count']


# In[39]:


fig = px.pie(accident_counts, names='accident', values='count',
             title='Distribution of Accident Reports',
             color='count'
)

fig.update_layout(
    width=700,  
    height=700,
    legend_title='Accident Type'  
)

fig.show()


# In[40]:


cleantitle_counts = train['clean_title'].value_counts().reset_index()
cleantitle_counts.columns = ['clean_title', 'count']


# In[41]:


fig = px.pie(cleantitle_counts, names='clean_title', values='count',
             title='Distribution of Clean Title',
             color='count'
)

fig.update_layout(
    width = 500,  
    height = 500,
    legend_title='clean_title'  
)

fig.show()


# # 2.2 Numerical Features

# In[42]:


numerical_columns = ['model_year', 'milage', 'price']


# In[43]:


for column in numerical_columns:
    plt.figure()
    plt.hist(train[column], bins=10, edgecolor='black')
    plt.title(f'{column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()


# In[45]:


plt.figure(figsize=(12, 6))
sns.barplot(x='brand', y='price', data=train, errorbar=None)
plt.title('Average Price by Car Brand')
plt.xlabel('Brand')
plt.ylabel('Average Price')
plt.xticks(rotation=90)  
plt.show()


# In[46]:


plt.figure(figsize=(12, 6))
sns.barplot(x='model_year', y='price', data=train, errorbar=None)
plt.title('Average Price by Model Year')
plt.xlabel('Model Year')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
plt.show()


# In[47]:


plt.figure(figsize=(14, 6))
sns.barplot(x='transmission', y='price', data=train, errorbar=None)
plt.title('Average Price by Transmission Type')
plt.xlabel('Transmission')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
plt.show()


# In[ ]:




