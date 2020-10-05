#!/usr/bin/env python
# coding: utf-8

# # In this project we will be performing analysis of Meterological Data
# # So Lets Start
# # First we will be importing the required Libraries which will be required in our project

# In[18]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Importing our dataset

# In[19]:


df = pd.read_csv('weatherHistory.csv')


# # Now Let's have a look at our dataset

# In[20]:


df.head()


# In[21]:


df.shape


# As you can see that our dataset is having a total of 96453 enteries and is having 12 different columns

# In[22]:


df.dtypes


# Before visulaisation we need to make date features -> date time object . For this we use to_datetime() function

# In[23]:


df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
df['Formatted Date']


# In[24]:


df.dtypes


# Now let's make Formatted Date our index of the dataset 

# In[25]:


df = df.set_index('Formatted Date')
df.head()


# As you can see the index of our dataset is now the formatted Date column as it will help us in further analysis

# Now since we have been given hourly data,we need to resample it monthly. Resampling is basically used for frequency conversion

# In[57]:


data_columns = ['Apparent Temperature (C)', 'Humidity']
df_mean = df.resample('MS').mean()
df_mean.head()


# Here "MS" denotes: Month starting We are displaying the average apparent temperature and humidity using mean() function

# # Visualising the data

# Now let's plot the variation in Apparent Temperature and Humidity with time
# lets import one more library i.e seaborn library

# In[58]:


import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
plt.figure(figsize=(14,6))
plt.title("Variation in Apparent Temperature and Humidity with time")
sns.lineplot(data=df_mean[data_columns])


# From Visualing the above plot we can say that the Apparent Temperature varies over time but its average is almost same as peaks lie on the same line and the Humidity is constant throughout the years

# 

# In[ ]:





# Now let's see the variation of windspeed with time

# In[45]:



warnings.filterwarnings("ignore")
plt.figure(figsize=(14,6))
plt.title("Variation in Windspeed(km) with time")
sns.lineplot(data=df_mean['Wind Speed (km/h)'])


# By seeing this plot we can say that the wind speed has varied a lot with time

# Now let's have a look at the variation in pressure with time

# In[ ]:





# In[46]:


warnings.filterwarnings("ignore")
plt.figure(figsize=(14,6))
plt.title("Variation in Pressure with time")
sns.lineplot(data=df_mean['Pressure (millibars)'])


# As you can see from the above plot there is a stiff decline in the pressure(millibar) at the end of the year 2014

# Now Let's have a look at the visibility throughout these years

# In[47]:


warnings.filterwarnings("ignore")
plt.figure(figsize=(14,6))
plt.title("Variation in Visibility with time")
sns.lineplot(data=df_mean['Visibility (km)'])


# As you can see  from the plot that the average visibility till the year end of 2012 is same as the peaks align in the same line but there is an increase in visibility after 2012

# 

# In[ ]:





# 

# Now lets have a look ka the varitaion in wind bearing

# In[52]:


warnings.filterwarnings("ignore")
plt.figure(figsize=(14,6))
plt.title("Variation in windbearing with time")
sns.lineplot(data=df_mean['Wind Bearing (degrees)'])


# In[60]:


#Plotting the variation in Apparent Temperature and Humidity for the month of April every year:

df1 = df_mean[df_mean.index.month==4]
print(df1)
df1.dtypes


# In[70]:


import matplotlib.dates as mdates
fig, ax = plt.subplots(figsize=(15,5))
ax.plot(df1.loc['2006-04-01':'2016-04-01', 'Apparent Temperature (C)'], marker='o', linestyle='-',label='Apparent Temperature (C)')
ax.plot(df1.loc['2006-04-01':'2016-04-01', 'Humidity'], marker='o', linestyle='-',label='Humidity')
ax.legend(loc = 'center right')
ax.set_xlabel('Month of April')


# Observation : No change in average humidity. Increase in average apparent temperature can be seen in the year 2009 then again it dropped in 2010 then there was a slight increase in 2011 then a significant drop is observed in 2015 and again it increased in 2016 

# Now lets have a look at rest of the data in the month of April

# In[75]:


fig, ax = plt.subplots(figsize=(15,5))
ax.plot(df1.loc['2006-04-01':'2016-04-01', 'Wind Speed (km/h)'], marker='o', linestyle='-',label='Wind Speed')

ax.legend(loc = 'center right')
ax.set_xlabel('Month of April')


# The plot of the wind is basically zig-zag but on an avergae we can say that the wind speed has increased in the month of April in the span of 10 years

# In[72]:


fig, ax = plt.subplots(figsize=(15,5))
ax.plot(df1.loc['2006-04-01':'2016-04-01', 'Visibility (km)'], marker='o', linestyle='-',label='Visibility')

ax.legend(loc = 'center right')
ax.set_xlabel('Month of April')


# The visiblity in the month of April has also increased in these 10 years

# Now lets have a look on pressure in the month of April in these 10 years

# In[73]:


fig, ax = plt.subplots(figsize=(15,5))
ax.plot(df1.loc['2006-04-01':'2016-04-01', 'Pressure (millibars)'], marker='o', linestyle='-',label='Pressure (millibars)')

ax.legend(loc = 'center right')
ax.set_xlabel('Month of April')


# As we can se that there was a significant drop in pressure during April 2011

# In[ ]:




