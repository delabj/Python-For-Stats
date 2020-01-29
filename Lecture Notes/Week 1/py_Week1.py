#!/usr/bin/env python
# coding: utf-8

# # First look - Python working on a dataset
# 
# ## Review Example Dataset: Home Loan applications 
# 
# Reference: Wooldridge, J. M. (2012). Introductory Econometrics: A Modern Approach (5th ed.)

# In[ ]:


#Importing Packages 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt #for plots
import seaborn as sns #for plots

#Setting up asethetics: how the answers are printed, I just want upto 4 decimal points. 
pd.set_option("display.float_format", lambda x: "%.4f" % x) 
np.set_printoptions(precision=4, suppress=True)


# ## Reading the Data

# In[ ]:


source_dir = "C:\\Users\\merta\\documents\\Python For Stats\\Week 1\\" #path of the file on my local computer 

data_file = pd.read_csv(source_dir +"loanapp.csv", #path + name of excel file
                    header=0) #which row gives the column names; 
                    #Python counter starts from 0, so header = 1 means row 2 in excel
                     
# We just need to use one column in the data, therefore we will rename that 
# column as returns. 
data = data_file[['loanamt', 'married', 'atotinc', 'price', 'apr', 'pubrec', 'male', 'mortno']]
data


# ## Descriptive Statistics
# 
# 

# In[5]:


data.describe(include='all')


# In[6]:


data.describe(include='all', percentiles = [0.85]) #I want to generate a specific percentile


# In[7]:


sns.distplot(data['loanamt'], kde=True); #with kernel density plot 


# In[8]:


from scipy.stats import norm # should import in Preamble

sns.distplot(data['loanamt'], fit=norm, kde=False); # fit normal distribution plots


# ## Predictive Model 1 - Linear Regression Model
# Predict Loan Amount in thousands (variable name - loanamt; continuous).
# We wil use four explantory variables: 
# 1. Married (Variable name - married; categorical = 1 if married)
# 2. Total monthly income of the applicant (Variable name - atotinc; continuous)
# 3. Purchase price in thousand (Variable name - price; continuous)
# 4. Appraised value in thousand (Variable name - apr; continuous)
# 
# There are many different combinations of explantory variables. Apart from domain knwoledge we will learn about more structural way of selecting explantory variables - a.k.a. Feature Selection in Python.

# In[9]:


from statsmodels.formula.api import ols # for predictive model 1 - continuous dependent varible 
from statsmodels.stats.anova import anova_lm

results = ols('loanamt ~ married + atotinc + price + apr', data).fit()
results.summary()


# In[10]:


# Hypothesis Tests
t_test = results.t_test('atotinc = 4000')
t_test


# In[11]:


# Generate ANOVA
table = anova_lm(results, typ=1) 
table


# ## Predictive Model 2 - Classification Model
# Predict whether a borrower will file for bankruptcy using four explantory variables:
# We will again use four explantory variables: 
# 1. Gender (Variable name - male; categorical = 1 if male)
# 2. Total monthly income of the applicant (Variable name - atotinc; continuous)
# 3. Purchase price in thousand (Variable name - price; continuous)
# 4. Mortagage history (Variable name - mortno; categorical = 1 if no  mortgage) 

# In[12]:


# review the data
print(data['pubrec'].value_counts())


# In[13]:


import statsmodels.api as sm #for predictive model 2 - categorical dependent varible 

# setting the predictive model 
y = data['pubrec']
X = data[['male', 'atotinc', 'price', 'mortno']]

results = sm.Logit(y,X).fit()
results.summary()


# ## No Dependent Variable? Clustering
# 
# Let's review another modelling method. Here we do not want to predict a characterstic of borrowrs, where as we want to form groups of individuals with similar characterstics. This technqiue is a part of Unsupervised Learning - Clsutering. 
# 
# Keeping things simple, I will use three variables from our data to form groups.  
# 1. Loan amount (Variable name - loanamt; continuous)
# 2. Total monthly income of the applicant (Variable name - atotinc; continuous)
# 3. Purchase price in thousand (Variable name - price; continuous)

# In[14]:


from sklearn.cluster import KMeans
from sklearn.utils import check_random_state

cluster_data = data[['loanamt', 'atotinc', 'price']]

results = KMeans(n_clusters=8, random_state=check_random_state(42)).fit(cluster_data)

# List for all cluster labels
cluster_labels = pd.DataFrame(results.labels_.astype(int), columns = ['Clusters'])
scatter_data = cluster_data.join(cluster_labels, how='inner')
scatter_data.head()


# ## Displaying the Results in a 3D plot 

# In[15]:


# 3D Scatter plots
from mpl_toolkits.mplot3d import Axes3D

plt.close('all')
fig = plt.figure(figsize=(15, 10))
fig.suptitle("Clusters of borrowers", fontsize=16)

ax = fig.add_subplot(111, projection='3d')
ax.scatter(scatter_data.iloc[:, 0], scatter_data.iloc[:, 1], scatter_data.iloc[:, 2],
                     c=scatter_data['Clusters'], cmap='plasma', s=10**1.5)

# Managing the aesthetics
ax.set_xlabel(data.columns[0], fontsize=14)
ax.set_ylabel(data.columns[1], fontsize=14)
ax.set_zlabel(data.columns[2], fontsize=14)
ax.xaxis.labelpad = 15
ax.yaxis.labelpad = 15
ax.zaxis.labelpad = 15

plt.show()
plt.clf() #clear the memory


# In[ ]:




