
# coding: utf-8

# # Reddit Predictor - DS Task

# Making a rank/score predictor based on the dataset with Cross validation.
# 
# 

# # Installing a package in python

# In[2]:

import pip    
def install(package):
   pip.main(['install', package])

# install('package_name')



# # Loading required packages

# In[3]:

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn; # Seaborn is a library for making attractive and informative statistical graphics in Python
from sklearn.linear_model import LinearRegression
import pylab as pl

seaborn.set()


# # Getting the current working directory and changing the directory as well.

# In[4]:

import os
print os.getcwd()
os.chdir('C://users/Shivam Panchal/Documents')


# In[5]:

# Importing the Data
import pandas as pd 
data = pd.read_csv('AmericanPolitics.csv')


# # Exploration of Data

# In[6]:

list(data.columns)


# In[7]:

data.head(10)


# In[8]:

data.describe()


# # Exploring the relationship between various features and output features.

# # Feature Selection
# 
# How do I decide which features to include in a linear model? Here's one idea:
# 
# Try different models, and only keep predictors in the model if they have small p-values.
# 
# Check whether the R-squared value goes up when you add new predictors.
# 
# What are the drawbacks to this approach?
# 
# Linear models rely upon a lot of assumptions (such as the features being independent), and if those assumptions are violated (which they usually 
# are), R-squared and p-values are less reliable.
# 
# Using a p-value cutoff of 0.05 means that if you add 100 predictors to a model that are pure noise, 5 of them (on average) will still be counted as significant.
# 
# R-squared is susceptible to overfitting, and thus there is no guarantee that a model with a high R-squared value will generalize. Below is an example:

# In[9]:

fig = plt.figure()
plot1 = fig.add_subplot(1,1,1)
plot1.scatter(data['score'],data['ups'])
plt.title('Score versus Upvotes')
plt.xlabel('Upvotes')
plt.ylabel('Score')
plt.show()


# In[10]:

fig = plt.figure()
plot2 = fig.add_subplot(1,1,1)
plot2.scatter(data['score'],data['downs'])
plt.title('Score versus Downvotes')
plt.xlabel('Downvotes')
plt.ylabel('Score')
plt.show()


# In[11]:

fig = plt.figure()
plot3 = fig.add_subplot(1,1,1)
plot3.scatter(data['score'],data['num_comments'])
plt.title('Score versus Number of Comments')
plt.xlabel('Number of Comments')
plt.ylabel('Score')
plt.show()


# In[12]:

# visualize the relationship between the features and the response using scatterplots
seaborn.pairplot(data, x_vars=['ups','downs','num_comments'], y_vars='score', size=7, aspect=0.7, kind='reg')


# In the given dataset, we came to the conclusion that, our target feature i.e. SCORE is dependent upon some variables and independent of some varialbles, #'ups' is a strong variable to preddict the score feature.
# 
# 'downs' and 'num_comments' are not strong features and did not hold any strong relationship for predicting the score feature, but they were assumed to be consisting of a good relation, buts its not true, as can be seen from above graph.
# 
# 
# Nw, there are variables such as 
# 'created_utc', 
# 'domain', 
# 'id', 
# 'title', 
# 'author', 
# 'permalink', 
# 'selftext', 
# 'link_flair_text', 
# 'thumbnail', link_flair_css_class',
# 'author_flair_css_class',
# 's_self',
# 'name',
# 'url', 
# 'distinguished',
# 'subreddit_id', 
# 
# these have no significant relation in predicting the score feature.
# 
# Many of them are even empty columns.
# 
# Now, categorical variables like, 'over_18' and 'edited' are also same foe all the posts, so they will also have no significance in predicting the sore feature.
# 
# 
# The only variable we are left with is 'ups' , which is a strong feature in predicting the score.
# 
# 
# 

# In[13]:

feature_cols = ['ups','downs','num_comments']
X = data[feature_cols]
X.head(10)


# # Agenda
# 
# Using the **train/test split** procedure for model evaluation?
# How does **K-fold cross-validation** overcome this limitation?
# How can cross-validation be used for selecting **tuning parameters**, choosing between **models**, and selecting **features**?
# What are some possible **improvements** to cross-validation?

# In[14]:

print(type(X))
print(X.shape)


# In[15]:

y = data['score']
y.head()


# In[16]:

print(type(y))
print(y.shape)


# # Cross validation for the dataset

# In[17]:

from sklearn.cross_validation import cross_val_score


# In[18]:

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# In[22]:

# use train/test split with different random_state values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)

# check classification accuracy of KNN with K=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# In[23]:

# simulate splitting a dataset of 25 observations into 5 folds
from sklearn.cross_validation import KFold
kf = KFold(25, n_folds=5, shuffle=False)

# print the contents of each training and testing set
print('{} {:^61} {}'.format('Iteration', 'Training set observations', 'Testing set observations'))
for iteration, data in enumerate(kf, start=1):
    print('{:^9} {} {:^25}'.format(iteration, data[0], data[1]))


# In[24]:

from sklearn.cross_validation import cross_val_score


# In[26]:

# 10-fold cross-validation with K=5 for KNN (the n_neighbors parameter)
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores)


# In[27]:

# use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean())


# In[28]:

# search for an optimal value of K for KNN
k_range = list(range(1, 31))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)


# In[29]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')


# # Splitting the dataset into training and testin model

# In[30]:

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# In[31]:

# default split is 75% for training and 25% for testing
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# # Developing the linear predictive model.

# In[32]:

# import model
from sklearn.linear_model import LinearRegression

# instantiate
linreg = LinearRegression()

# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, y_train)


# In[34]:

# print the intercept and coefficients
print(linreg.intercept_)
print(linreg.coef_)


# In[35]:

# pair the feature names with the coefficients
list(zip(feature_cols, linreg.coef_))


# In[36]:

# make predictions on the testing set
y_pred = linreg.predict(X_test)


# In[37]:

y_pred


# In[39]:

# compute the RMSE of our predictions
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:



