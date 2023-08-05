#!/usr/bin/env python
# coding: utf-8

# # TASK 2 : TITANIC CLASSIFICATION

# ## Algorithm which tells whether the person will be save from sinking or not

# # Purpose
# **Titanic** is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the **Titanic** sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.
# 
# In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).
# 
# The dataset is available at Kaggle : https://www.kaggle.com/datasets/rahulsah06/titanic

# ## STEPS INVOLVED : 
# ### 1. Problem understanding and definition
# ### 2. Data Loading and Importing the necessary libraries
# ### 3. Data understanding using Exploratory Data Analysis (EDA)
# ### 4. Feature Engineering and Data Processing
# ### 5. Feature Engineering and Data Processing
# ### 6. Model Evaluation

# <a id="section1"></a>
# ## 1. Problem understanding and definition
# 
# 
# In this challenge, we need to complete the __analysis__ of what sorts of people were most likely to __survive__. In particular,  we apply the tools of __machine learning__ to predict which passengers survived the tragedy
# 
# - Predict whether passenger will __survive or not__.

# <a id="section2"></a>
# ## 2. Data Loading and Importing the necessary libraries

# In[2]:


# Linear algebra
import numpy as np

# Data manipulation and analysis
import pandas as pd

# Data visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import style

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# <a id="section201"></a>
# ### 2.1  Loading the data files 
# 
# Here we import the data. For this analysis, we will be exclusively working with the Training set. We will be validating based on data from the training set as well. For our final submissions, we will make predictions based on the test set.

# In[3]:


train_df = pd.read_csv('task_2 Train.csv')
test_df = pd.read_csv('task_2 Test.csv')

train_df['train_test'] = 1
test_df['train_test'] = 0
# test_df['Survived'] = np.NaN
all_data = pd.concat([train_df,test_df])

get_ipython().run_line_magic('matplotlib', 'inline')
all_data.columns


# In[4]:


train_df.head(10)


# In[5]:


test_df.head(10)


# <a id="section202"></a>
# ### 2.2 About The Dataset

# The data has been split into two groups:
# - training set (train.csv)
# - test set (test.csv)
# 
# The training set includes passengers survival status (also know as the ground truth from the titanic tragedy) which along with other features like gender, class, fare and pclass is used to create the machine learning model.
# 
# The test set should be used to see how well the model performs on unseen data. The test set does not provide passengers survival status. We are going to use our model to predict passenger survival status.
# 
# This is clearly a <font color='red'>__Classification problem__.</font> In predictive analytics, when the <font color='red'>__target__</font> is a categorical variable, we are in a category of tasks known as <font color='red'>__classification tasks.__</font>

# | Column Name          | Description                                                | Key                    |
# | ---------------------| ---------------------------------------------------------- | ---------------------- |
# | __PassengerId__      | Passenger Identity                                         |                        | 
# | __Survived__         | Whether passenger survived or not                          | 0 = No, 1 = Yes        | 
# | __Pclass__           | Class of ticket, a proxy for socio-economic status (SES)| 1 = 1st, 2 = 2nd, 3 = 3rd | 
# | __Name__             | Name of passenger                                          |                        | 
# | __Sex__              | Sex of passenger                                           |                        |
# | __Age__              | Age of passenger in years                                  |                        |
# | __SibSp__            | Number of sibling and/or spouse travelling with passenger  |                        |
# | __Parch__            | Number of parent and/or children travelling with passenger |                        |
# | __Ticket__           | Ticket number                                              |                        |
# | __Fare__             | Price of ticket                                            |                        |
# | __Cabin__            | Cabin number                                               |                        |
# | __Embarked__         | Port of embarkation                                        | C = Cherbourg, Q = Queenstown, S = Southampton |

# <a id="section3"></a>
# ## 3. Data understanding using Exploratory Data Analysis (EDA)
# __Exploratory Data Analysis__ refers to the critical process of performing initial investigations on data so as to discover patterns, to spot anomalies, to test hypothesis and to check assumptions with the help of summary statistics and graphical representations.
# 
# In summary, it's an approach to analyzing data sets to summarize their main characteristics, often with visual methods.

# In[6]:


train_df.info()


# The training-set has 891 rows and 11 features + the __target variable (survived).__ 2 of the features are floats, 5 are integers and 5 are objects.

# In[7]:


train_df.describe()


# #### Conclusions from .describe() method
# __.describe()__ gives an understanding of the central tendencies of the numeric data.
# 
# - Above we can see that __38% out of the training-set survived the Titanic.__ 
# - We can also see that the passenger age range from __0.4 to 80 years old.__
# - We can already detect some features that contain __missing values__, like the ‘Age’ feature (714 out of 891 total).
# - There's an __outlier__ for the 'Fare' price because of the differences between the 75th percentile, standard deviation, and the max value (512). We might want to drop that value.

# <a id="section301"></a>
# ### 3.1 Exploring missing data

# In[8]:


total = train_df.isnull().sum().sort_values(ascending=False)
percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(13)


# The __'Embarked'__ feature has only 2 missing values, which can easily be filled or dropped. It will be much more tricky to deal with the __‘Age’__ feature, which has 177 missing values. The __‘Cabin’__ feature needs further investigation, but it looks like we might want to drop it from the dataset since 77% is missing.

# In[9]:


train_df.columns.values


# Above we can see the 11 features and the target variable (survived). __What features could contribute to a high survival rate ?__
# 
# I believe it would make sense if everything except ‘PassengerId’, ‘Name’ and ‘Ticket’ would be high correlated with survival rate.

# <a id="section302"></a>
# ### 3.2 Dealing with the outlier

# In[10]:


sns.boxplot(x='Survived',y='Fare',data=train_df);


# #### Passengers who paid over 300

# In[11]:


train_df[train_df['Fare']>300]


# #### Drop the outliers
# 
# It might be beneficial to drop those outliers for the model. Further investigation needs to be done.

# In[12]:


# train_df = train_df[train_df['Fare']<300]


# #### The Captain went down with the ship
# __"The captain goes down with the ship"__ is a maritime tradition that a sea captain holds ultimate responsibility for both his/her ship and everyone embarked on it, and that in an emergency, he/she will either save them or die trying.
# 
# In this case, __Captain Edward Gifford Crosby__ went down with Titanic in a heroic gesture trying to save the passengers.

# In[13]:


train_df[train_df['Name'].str.contains("Capt")]


# <a id="section303"></a>
# ### 3.3 Embarked, Pclass and Sex:

# In[14]:


FacetGrid = sns.FacetGrid(train_df, col='Embarked', height=4, aspect=1.2)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci=95.0, palette='deep', order=None, hue_order=None)
FacetGrid.add_legend();


# <a id="section304"></a>
# ### 3.4 Distribution of Pclass and Survived

# In[15]:


sns.set(style='darkgrid')
plt.subplots(figsize = (8,6))
ax=sns.countplot(x='Sex', data = train_df, hue='Survived', edgecolor=(0,0,0), linewidth=2)

# Fixing title, xlabel and ylabel
plt.title('Passenger distribution of survived vs not-survived', fontsize=25)
plt.xlabel('Gender', fontsize=15)
plt.ylabel("# of Passenger Survived", fontsize = 15)
labels = ['Female', 'Male']

# Fixing xticks.
plt.xticks(sorted(train_df.Survived.unique()),labels);


# In[16]:


train_df.groupby(['Sex']).mean()


# As previously mentioned, women are much more likely to survive than men. __74% of the women survived, while only 18% of men survived.__

# ### Looking deeper into differences between females and males statistics

# In[17]:


train_df.groupby(['Sex','Pclass']).mean()


# We are grouping passengers based on Sex and Ticket class (Pclass). Notice the difference between survival rates between men and women.
# 
# Women are much more likely to survive than men, **specially women in the first and second class.** It also shows that men in the first class are almost **3-times more likely to survive** than men in the third class.

# <a id="section305"></a>
# ### 3.5 Age and Sex distributions

# In[18]:


survived = 'survived'
not_survived = 'not survived'

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

women = train_df[train_df['Sex']=='female']
men = train_df[train_df['Sex']=='male']

# Plot Female Survived vs Not-Survived distribution
ax = sns.histplot(women[women['Survived']==1].Age.dropna(), bins=20, label = survived, ax = axes[0],color='b', kde=True)
ax = sns.histplot(women[women['Survived']==0].Age.dropna(), bins=20, label = not_survived, ax = axes[0],color='r', kde=True)
ax.legend()
ax.set_title('Female')

# Plot Male Survived vs Not-Survived distribution
ax = sns.histplot(men[men['Survived']==1].Age.dropna(), bins=20, label = survived, ax = axes[1],color='b', kde=True)
ax = sns.histplot(men[men['Survived']==0].Age.dropna(), bins=20, label = not_survived, ax = axes[1],color='r', kde=True)
ax.legend()
ax.set_title('Male');


# We can see that __men__ have a higher probability of survival when they are between __18 and 35 years old.__ For __women,__ the survival chances are higher between __15 and 40 years old.__
# 
# For men the probability of survival is very low between the __ages of 5 and 18__, and __after 35__, but that isn’t true for women. Another thing to note is that __infants have a higher probability of survival.__

# ### Saving children first

# In[19]:


train_df[train_df['Age']<18].groupby(['Sex','Pclass']).mean()


# __Children below 18 years of age__ have higher chances of surviving, proven they saved childen first

# <a id="section306"></a>
# ### 3.6 Passenger class distribution; Survived vs Non-Survived

# In[20]:


plt.subplots(figsize = (8,8))
ax=sns.countplot(x='Pclass',hue='Survived',data=train_df)
plt.title("Passenger Class Distribution - Survived vs Non-Survived", fontsize = 25);


# In[21]:


plt.subplots(figsize=(10,8))
ax=sns.kdeplot(train_df.loc[(train_df['Survived'] == 0),'Pclass'],shade=True,color='r',label='Not Survived')
ax.legend()
ax=sns.kdeplot(train_df.loc[(train_df['Survived'] == 1),'Pclass'],shade=True,color='b',label='Survived')
ax.legend()

plt.title("Passenger Class Distribution - Survived vs Non-Survived", fontsize = 25)
labels = ['First', 'Second', 'Third']
plt.xticks(sorted(train_df.Pclass.unique()),labels);


# In[22]:


plt.subplots(figsize = (8,6))
sns.barplot(x='Pclass', y='Survived', data=train_df);
plt.title("Passenger Class Distribution - Survived Passengers", fontsize = 25);


# The graphs above clearly shows that __economic status (Pclass)__ played an important role regarding the potential survival of the Titanic passengers. First class passengers had a much higher chance of survival than passengers in the 3rd class. We note that:
# 
# - 63% of the 1st class passengers survived the Titanic wreck
# - 48% of the 2nd class passengers survived
# - Only 24% of the 3rd class passengers survived

# <a id="section307"></a>
# ### 3.7 Correlation Matrix and Heatmap

# In[23]:


# Look at numeric and categorical values separately 
df_num = train_df[['Age','SibSp','Parch','Fare']]
df_cat = train_df[['Survived','Pclass','Sex','Ticket','Cabin','Embarked']]


# In[24]:


plt.subplots(figsize = (12,6))
sns.heatmap(df_num.corr(), annot=True,cmap="RdBu")
plt.title("Correlations Among Numeric Features", fontsize = 18);


# We notice from the heatmap above that:
# - __Parents and sibling like to travel together <font color='blue'>(light blue squares)__</font>
# - __Age has a high negative correlation with number of siblings__

# <a id="section4"></a>
# ## 4. Feature Engineering and Data Processing
# __Feature Engineering__ is the process of using raw data to create features that will be used for predictive modeling. Using, transforming, and combining existing features to define new features are also considered to be feature engineering.

# <a id="section401"></a>
# ### 4.1 Drop 'PassengerId'
# 
# First, I will drop ‘PassengerId’ from the train set, because it does not contribute to a persons' survival probability. I will not drop it from the test set, since it is required for the submission.

# In[25]:


train_df = train_df.drop(['PassengerId'], axis=1)
train_df.head()


# <a id="section402"></a>
# ### 4.2 Combining SibSp and Parch
# 
# SibSp and Parch would make more sense as a combined feature that shows the total number of relatives a person has on the Titanic. I will create the new feature 'relative' below, and also a value that shows if someone is not alone.

# In[26]:


data = [train_df, test_df]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)
train_df['not_alone'].value_counts()


# In[27]:


plt.subplots(figsize = (16,4))
ax = sns.lineplot(x='relatives',y='Survived', data=train_df)


# <a id="section403"></a>
# ### 4.3 Missing Data
# 
# As a reminder, we have to deal with __Cabin (687 missing values), Embarked (2 missing values)__ and __Age (177 missing values).__

# In[28]:


import re
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
data = [train_df, test_df]

for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)


# In[29]:


# We can now drop the Cabin feature
train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)


# #### Age
# 
# As seen previously on __"3.1 Dealing with Missing Values"__, there are a lot of missing 'Age' values (177 data points). We can normalize the 'Age' feature by creating an array that contains random numbers, which are computed based on the mean age value in regards to the standard deviation and is_null.

# In[30]:


data = [train_df, test_df]

for dataset in data:
    mean = train_df["Age"].mean()
    std = test_df["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    
    # Compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    
    # Fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_df["Age"].astype(int)


# In[31]:


train_df["Age"].isnull().sum()


# #### Embarked
# 
# Since the Embarked feature has only 2 missing values, we will fill these with the most common one.

# In[32]:


train_df['Embarked'].describe()


# We notice the most popular embark location is __Southampton (S).__

# In[33]:


common_value = 'S'
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)


# In[34]:


train_df['Embarked'].isnull().sum()


# <a id="section404"></a>
# ### 4.4 Converting Features

# In[35]:


train_df.info()


# We can see that __'Fare'__ is a float data-type. Also, we need to deal with 4 categorical features: __Name, Sex, Ticket, and Embarked__

# #### Fare
# 
# Converting 'Fare' from __float64__ to __int64__ using the __astype()__ function provided by pandas

# In[36]:


data = [train_df, test_df]

for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)


# In[37]:


train_df.info()


# #### Name
# 
# Feature Engineering the name of passengers to extract a person's title (Mr, Miss, Master, and Other), so we can build another feature called **'Title'** out of it.

# In[38]:


data = [train_df, test_df]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}

for dataset in data:
    # Extract titles
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
    
    # Replace titles with a more common title or as Other
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr','Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
    # Convert titles into numbers
    dataset['Title'] = dataset['Title'].map(titles)
    
    # Filling NaN with 0 just to be safe
    dataset['Title'] = dataset['Title'].fillna(0)


# In[39]:


train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)


# In[40]:


# Checking results
train_df.head()


# #### Sex
# 
# Convert feature 'Sex' into numeric values
# - male = 0
# - female = 1

# In[41]:


genders = {"male": 0, "female": 1}
data = [train_df, test_df]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)


# In[42]:


train_df.head()


# #### Ticket

# In[43]:


train_df['Ticket'].describe()


# Since the __'Ticket'__ feature has 681 unique values, it would be very hard to convert them into an useful feature. __Hence, we will drop it from the DataFrame.__

# In[44]:


train_df = train_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1)


# In[45]:


train_df.head()


# #### Convert 'Embarked' feature into numeric values

# In[46]:


ports = {"S": 0, "C": 1, "Q": 2}
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)


# In[47]:


train_df.head()


# <a id="section405"></a>
# ### 4.5 Creating new Categories

# In[48]:


data = [train_df, test_df]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6


# In[49]:


# Checking the distribution
train_df['Age'].value_counts()


# #### Fare
# 
# For the 'Fare' feature, we need to do the same as with the 'Age' feature. But it isn't that easy, because if we cut the range of the fare values into a few equally big categories, 80% of the values would fall into the first category. Fortunately, we can use pandas "qcut()" function, that we can use to see, how we can form the categories.

# In[50]:


train_df.head()


# In[51]:


pd.qcut(train_df['Fare'], q=6)


# #### Using the values from **pd.qcut()** to create bins for Fare

# In[52]:


data = [train_df, test_df]

for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7) & (dataset['Fare'] <= 8), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 8) & (dataset['Fare'] <= 14), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 14) & (dataset['Fare'] <= 26), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 26) & (dataset['Fare'] <= 52), 'Fare']   = 4
    dataset.loc[dataset['Fare'] > 52, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)


# In[53]:


# Checking the dataset
train_df.head(10)


# <a id="section5"></a>
# ## 5. Model building

# In[54]:


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()


# <a id="section501"></a>
# ### 5.1 Stochastic Gradient Descent (SGD)

# In[55]:


sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

sgd.score(X_train, Y_train)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

# Print score
print(round(acc_sgd,2,), "%")


# <a id="section502"></a>
# ### 5.2 Decision Tree

# In[56]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

# Print score
print(round(acc_decision_tree,2,), "%")


# <a id="section503"></a>
# ### 5.3 Random Forest

# In[57]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

# Print score
print(round(acc_random_forest,2,), "%")


# <a id="section504"></a>
# ### 5.4 Logistic Regression

# In[58]:


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

# Print score
print(round(acc_log,2,), "%")


# <a id="section505"></a>
# ### 5.5 KNN

# In[59]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

# Print score
print(round(acc_knn,2,), "%")


# <a id="section506"></a>
# ### 5.6 Gaussian Naive Bayes

# In[60]:


gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

# Print score
print(round(acc_gaussian,2,), "%")


# <a id="section507"></a>
# ### 5.7 Perceptron

# In[61]:


perceptron = Perceptron(max_iter=1000)
perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

# Print score
print(round(acc_perceptron,2,), "%")


# <a id="section6"></a>
# ## 6. Model evaluation
# 
# ### Which one is the best model?

# In[62]:


results = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 
              'Decision Tree'],
    'Score': [acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_decision_tree]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)


# The __Random Forest classifier__ goes on top of the Machine Learning models, followed by **Decision Tree** and __KNN__ respectfully. Now we need to check how the Random Forest performs by using cross validation.

# <a id="section601"></a>
# ### 6.1 K-Fold Cross Validation
# K-Fold Cross Validation randomly splits the training data into __K subsets called folds__. Image we split our data into 4 folds (K = 4). The random forest model would be trained and validated 4 times, using a different fold for validation every time, while it would be trained on the remaining 3 folds.
# 
# The image below shows the process, using 4 folds (K = 4). Every row represents one training + validation process. In the first row, the model is trained on the second, third and fourth subsets and validated on the first subset. In the second row, the model is trained on the first, third and fourth subsets and validated on the second subset. K-Fold Cross Validation repeats this process until every fold acted once as an evaluation fold.

# In[63]:


from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")


# In[64]:


print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


# This looks much more realistic than before. The __Random Forest classifier__ model has an average __accuracy of 81%__ with a __standard deviation of 3.9%__. The standard deviation tell us how precise the estimates are.
# 
# - This means the accuracy of our model can differ __± 3.9%__ 
# 
# I believe the accuracy looks good. Since Random Forest is a model easy to use, we will try to increase its performance even further in the following section.

# <a id="section602"></a>
# ### 6.2 Random Forest

# In[65]:


importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')


# In[66]:


importances.head(12)


# In[67]:


importances.plot.bar();


# <a id="section604"></a>
# ### 6.4 Results
# 
# __'not_alone' and 'Parch' don't play a significant role in the Random Forest classifiers prediction process__. Thus, I will drop them from the DataFrame and train the classifier once again. We could also remove more features, however, this would inquire more investigations of the feature's effect on our model. For now, I will only remove 'not_alone' and 'Parch' from the DataFrame.

# In[68]:


# Dropping not_alone
train_df  = train_df.drop("not_alone", axis=1)
test_df  = test_df.drop("not_alone", axis=1)

# Dropping Parch
train_df  = train_df.drop("Parch", axis=1)
test_df  = test_df.drop("Parch", axis=1)


# In[69]:


# # Reassigning features
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()


# ### Training the Random Forest classifier once again

# In[70]:


random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

# Print scores
print(round(acc_random_forest,2,), "%")


# #### Feature importance without 'not_alone' and 'Parch' features

# In[71]:


importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')


# In[72]:


importances.head(12)


# The __Random Forest__ model predicts as good as it did before. A general rule is that, the more features you have, the more likely your model will suffer from overfitting and vice versa. But I think our data looks fine for now and hasn't too much features.
# 
# Moreover, there is another way to validate the Random Forest classifier, which is as accurate as the score used before. We can use something called __Out of Bag (OOB) score__ to estimate the generalization accuracy. __Basically, the OOB score is computed as the number of correctly predicted rows from the out of the bag sample__.

# In[73]:


print("oob score:", round(random_forest.oob_score_, 4)*100, "%")


# Now we can start tuning the **hyperameters** of random forest.

# <a id="section605"></a>
# ### 6.5 Hyperparameter Tuning

# In[85]:


# Simple performance reporting function
def clf_performance(classifier, model_name):
    print(model_name)
    print('Best Score: ' + str(classifier.best_score_))
    print('Best Parameters: ' + str(classifier.best_params_))


# In[86]:


from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier(random_state = 1)
param_grid =  {'n_estimators': [400,450,500,550],
               'criterion':['gini','entropy'],
               'bootstrap': [True],
               'max_depth': [15, 20, 25],
               'max_features': ['auto','sqrt', 10],
               'min_samples_leaf': [2,3],
               'min_samples_split': [2,3]}
                                  
clf_rf = GridSearchCV(rf, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_rf = clf_rf.fit(X_train,Y_train)

# Print score
clf_performance(best_clf_rf,'Random Forest')


# <a id="section606"></a>
# ### 6.6 Testing new parameters

# In[87]:


random_forest = RandomForestClassifier(criterion = "gini",
                                       max_depth = 20,
                                       max_features='auto',
                                       min_samples_leaf = 3, 
                                       min_samples_split = 2,
                                       n_estimators=450,
                                       oob_score=True, 
                                       random_state=1, 
                                       n_jobs=-1)

random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

print("oob score:", round(random_forest.oob_score_, 4)*100, "%")


# <a id="section607"></a>
# ### 6.7 Further evaluation

# #### Confusion Matrix

# In[88]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

predictions = cross_val_predict(random_forest, X_train, Y_train, cv=3)
confusion_matrix(Y_train, predictions)


# The first row is about the not-survived-predictions: __494 passengers were correctly classified as not survived__ (called true negatives) and __55 where wrongly classified as not survived__ (false positives).
# 
# The second row is about the survived-predictions: __98 passengers where wrongly classified as survived__ (false negatives) and __244 where correctly classified as survived__ (true positives).
# 
# A confusion matrix produces an idea of how accurate the model is.

# <a id="section608"></a>
# ### 6.8 Precision and Recall

# In[89]:


from sklearn.metrics import precision_score, recall_score

print("Precision:", precision_score(Y_train, predictions))
print("Recall:",recall_score(Y_train, predictions))


# Our model predicts correctly that __a passenger survived 81% of the time__ (precision). The __recall__ tells us that __71% of the passengers tested actually survived.__

# <a id="section609"></a>
# ### 6.9 F-score

# It is possible to combine precision and recall into one score, which is called the F-score. The F-score is computed with the harmonic mean of precision and recall. Note that it assigns more weight to low values. As a result, the classifier will only get a high F-score if both recall and precision are high.

# In[90]:


from sklearn.metrics import f1_score
f1_score(Y_train, predictions)


# There we have it, a __76% F-score.__ The score is not high because we have a recall of 71%. Unfortunately, the F-score is not perfect, because it favors classifiers that have a similar precision and recall. This can be a problem because often times we are searching for a high precision and other times a high recall. An increase of precision can result in a decrease of recall, and vice versa (depending on the threshold). This is called the __precision/recall trade-off.__

# <a id="section610"></a>
# ### 6.10 Precision Recall Curve

# For each person the Random Forest algorithm has to classify, it computes a probability based on a function and it classifies the person as __survived__ (when the score is bigger the than threshold) or as __not survived__ (when the score is smaller than the threshold). That’s why the threshold plays an important part in this process.
# 
# Let's plot the precision and recall with the threshold using matplotlib.

# In[91]:


from sklearn.metrics import precision_recall_curve

# Getting the probabilities of our predictions
y_scores = random_forest.predict_proba(X_train)
y_scores = y_scores[:,1]

precision, recall, threshold = precision_recall_curve(Y_train, y_scores)
def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])

plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()


# We can see in the graph above that the recall is falling of rapidly when the precision reaches around 85%. Thus, we may want to select the precision/recall trade-off before this point (maybe at around 75%).
# 
# Now we are able to choose a threshold, that gives the best precision/recall trade-off for the current problem. For example, if a precision of 80% is required, we can easily look at the plot and identify the threshold needed, which is around 0.4. Then we could train the model with exactly that threshold and expect the desired accuracy.
# 
# __Another way is to plot the precision and recall against each other:__

# In[92]:


def plot_precision_vs_recall(precision, recall):
    plt.plot(recall, precision, "b--", linewidth=3)
    plt.xlabel("precision", fontsize=19)
    plt.ylabel("recall", fontsize=19)
    plt.axis([0, 1.2, 0, 1.4])

plt.figure(figsize=(14, 7))
plot_precision_vs_recall(precision, recall)
plt.show()


# <a id="section611"></a>
# ### 6.11 ROC AUC Curve

# Another way to evaluate and compare binary classifiers is the ROC AUC Curve. This curve plots the true positive rate (also called recall) against the false positive rate (ratio of incorrectly classified negative instances), instead of plotting the precision versus the recall values.

# In[93]:


from sklearn.metrics import roc_curve

# Compute true positive rate and false positive rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, y_scores)

# Plotting them against each other
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=3, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()


# The red line represents a purely random classifier (e.g. a coin flip). Thus, the classifier should be as far away from it as possible. The Random Forest model looks good.
# 
# There's a tradeoff here because the classifier produces more false positives the higher the true positive rate is.

# <a id="section612"></a>
# ### 6.12 ROC AUC Score

# The ROC AUC Score is the corresponding score to the ROC AUC Curve. It is simply computed by measuring the area under the curve, which is called AUC.
# 
# A classifier that is 100% correct would have a ROC AUC Score of 1, and a completely random classifier would have a score of 0.5.

# In[94]:


from sklearn.metrics import roc_auc_score
r_a_score = roc_auc_score(Y_train, y_scores)
print("ROC-AUC-Score:", r_a_score)


# We got a __93% ROC AUC Score__
