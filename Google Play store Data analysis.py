#!/usr/bin/env python
# coding: utf-8

# In[121]:


import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)


# In[122]:


# Read in the dataset
apps = pd.read_csv("googleplaystore.csv")

# drop duplicates
apps.drop_duplicates(inplace=True)
apps.head()


# In[123]:


# Total number of apps 
print("Total number of apps in the dataset: ", apps.shape[0])


# In[124]:


# concise summary of apps dataframe
apps.info()


# The data has 12 object and 1 numeric feature i.e. Rating.

# ## Data Cleaning

# In[125]:


apps.sample(5)


# In[126]:


apps.isnull().sum()


# For analysis later we need to convert Installs and Price into numeric type, the presence of special characters (, $ +) in the Installs and Price columns make their conversion to a numerical data type difficult. So, they are needed to be removed.

# In[127]:


chars = ['+', ',', '$']
cols = ['Installs', 'Price']

for col in cols:
    for char in chars:
        apps[col] = apps[col].astype(str).str.replace(char, '')
    apps[col] = pd.to_numeric(apps[col], errors='ignore')


# It can be seen that data has metric prefixes (Kilo and Mega) along with another string. Replacing k and M with their values to convert values to numeric.

# In[128]:


apps['Size'] = apps['Size'].str.replace('k','e+3')
apps['Size'] = apps['Size'].str.replace('M','e+6')
apps['Size'].head()


# There is another string type of value in Size which is 'Varies with device' and one which is '1000+'.

# In[129]:


# replacing 'Varies with device' with nan
apps['Size'] = apps['Size'].replace('Varies with device', np.nan)

# Converting 1,000+ to 1000, to make it numeric
apps['Size'] = apps['Size'].replace('1,000+',1000)
apps['Size']


# In[130]:


# Converting 'Size' to numeric form
apps['Size'] = pd.to_numeric(apps['Size'])
apps['Size']


# In[131]:


# Checking for unique values of 'Price' and to check if there is any abnormality:
apps['Price'].unique()


# As we can see there is one value called 'Everyone' that needs to be removed.

# In[132]:


apps['Price'] = apps['Price'].replace('Everyone', np.nan)


# In[133]:


apps['Price'] = pd.to_numeric(apps['Price'])
apps['Price']


# ## Distribution of Apps in different categries

# In[134]:


# the total number of unique categories
num_categories = len(apps['Category'].unique())
print('Number of categories = ', num_categories)


# In[135]:


# Visualizing distribution of apps across various categories
num_apps_in_category = apps['Category'].value_counts().sort_values(ascending = False)

# Distribution of apps in different categories 

data = [go.Bar(
        x = num_apps_in_category.index, # index = category name
        y = num_apps_in_category.values, # value = count
)]

plotly.offline.iplot(data)


# From the above graph, we can see that Family and Game apps have the highest market prevalence. Interestingly, Tools, Business and Medical apps are also at the top. However, there is very low market for beauty, comics and parenting apps.

# ## Distribution of Apps Ratings

# In[136]:


# Average rating of apps
avg_app_rating = apps['Rating'].mean()
print('Average app rating = ', avg_app_rating)

# Distribution of apps according to their ratings
data = [go.Histogram(
        x = apps['Rating']
)]

# Vertical dashed line to indicate the average app rating
layout = {'shapes': [{
              'type' :'line',
              'x0': avg_app_rating,
              'y0': 0,
              'x1': avg_app_rating,
              'y1': 1000,
              'line': { 'dash': 'dashdot'}
          }]
          }

plotly.offline.iplot({'data': data, 'layout': layout})


# App ratings (on a scale of 1 to 5) impact the discoverability, conversion of apps as well as the company's overall brand image. Ratings are a key performance indicator of an app.
# 
# From our research, we found that the average ratings across all app categories is 4.17. The histogram plot above indicates that the majority of the apps are highly rated with only a few exceptions in the low-rated apps.

# ## Android version used

# In[137]:


# Count the number of apps belonging to each android version
num_apps_in_android = apps['Android Ver'].value_counts().sort_values(ascending = False)


data = [go.Bar(
        x = num_apps_in_android.index, # index = category name
        y = num_apps_in_android.values, # value = count
)]


plotly.offline.iplot(data)


# We can see that most of the apps are made for android version 4.1 and above. And most of the apps do not support either very old versions or latest versions.

# ## How Size and Price of an app affect ratings 

# In[138]:


sns.set_style("darkgrid")

# Filter rows where both Rating and Size values are not null
apps_with_size_and_rating_present = apps[(~apps['Rating'].isnull()) & (~apps['Size'].isnull())]

# Subset for categories with at least 250 apps
large_categories = apps_with_size_and_rating_present.groupby('Category').filter(lambda x: len(x) >= 250).reset_index()

# Plot size vs. rating
plt.scatter(large_categories['Size'], large_categories['Rating'])
plt.xlabel('Size')
plt.ylabel('Rating')
plt.title('Size vs Rating')
plt.show()


# As the size of the app increases, though the volume of rating decreases but higher rating is there for apps with more size.
# 
# We can say that the users prefer light-weighted apps over system heavy apps. We find that the majority of top rated apps (rating over 4) range from 2 MB to 20 MB.

# In[139]:


# Subset apps whose 'Type' is 'Paid'
paid_apps = apps_with_size_and_rating_present[apps_with_size_and_rating_present['Type'] == 'Paid']

# Plot price vs. rating
plt.scatter(paid_apps['Price'], paid_apps['Rating'])
plt.xlabel('Price of the app')
plt.ylabel('Rating')
plt.title('Price vs Rating')
plt.show()


# Rating is more for the apps which are free and very few of highly paid apps are rated. We also find that the vast majority of apps price themselves under $10.
# 
# Thus, we can say that free and lighter apps are given more ratings.

# ## How can companies make larger profits
# 
# The costs of apps are largely based on features, complexity, and platform. There are many factors to consider when selecting the right pricing strategy for your mobile app. It is important to consider the willingness of your customer to pay for your app.

# In[140]:


fig, ax = plt.subplots()
fig.set_size_inches(15, 8)

# Select a few popular app categories
popular_app_cats = apps[apps.Category.isin(['GAME', 'FAMILY', 'PHOTOGRAPHY',
                                            'MEDICAL', 'TOOLS', 'FINANCE',
                                            'LIFESTYLE','BUSINESS'])]

# Examine the price trend by plotting Price vs Category
plt.scatter(popular_app_cats['Price'], popular_app_cats['Category'])
plt.xlabel('Price of the app')
plt.ylabel('Categories')
plt.title('Price of the apps according to their categories')
plt.show()


# From above graph, we see that Family, Lifestyle and Finance apps are the most expensive, followed by medical apps.

# ## Popularity of paid apps vs free apps

# In[141]:


# Prep the data for a box plot that compares the number of installs of paid apps vs. number of installs of free apps.
trace0 = go.Box(
    # Data for paid apps
    y=apps[apps['Type'] == 'Paid']['Installs'],
    name = 'Paid'
)

trace1 = go.Box(
    # Data for free apps
    y=apps[apps['Type'] == 'Free']['Installs'],
    name = 'Free'
)

layout = go.Layout(
    title = "Number of downloads of paid apps vs. free apps",
    yaxis = dict(
        type = 'log',
        autorange = True
    )
)

# Add trace0 and trace1 to a list for plotting
data = [trace0, trace1]
plotly.offline.iplot({'data': data, 'layout': layout})


# It turns out that paid apps have a relatively lower number of installs than free apps, as expected.

# ## Sentiment Analysis

# In[142]:


# Load the user review data 
reviews_df = pd.read_csv('googleplaystore_user_reviews.csv')
reviews_df.head()


# In[143]:


# Join and merge the two dataframe
merged_df = pd.merge(apps, reviews_df, on = ['App'], how = "inner")
merged_df.head()


# In[144]:


# Drop NA values from Sentiment and Translated_Review columns
merged_df = merged_df.dropna(subset=['Sentiment', 'Translated_Review'])


# In[145]:


sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11, 8)

# User review sentiment polarity for paid vs. free apps
ax = sns.boxplot(x = 'Type', y = 'Sentiment_Polarity', data = merged_df)
ax.set_title('Sentiment Polarity Distribution')


# Mining user review data to determine how people feel about your product, brand, or service. By plotting sentiment polarity scores of user reviews for paid and free apps, we observe that free apps receive a lot of harsh comments, as indicated by the outliers on the negative y-axis. Reviews for paid apps appear never to be extremely negative. This may indicate something about app quality, i.e., paid apps being of higher quality than free apps on average. The median polarity score for paid apps is a little higher than free apps, thereby syncing with our previous observation.
