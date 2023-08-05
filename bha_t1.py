#!/usr/bin/env python
# coding: utf-8

# # DATA SCIENCE INTERN @BHARAT INTERN

# # TASK 1 : STOCK PREDICTION

# ## PURPOSE : TO PREDICT THE STOCK PRICE OF A COMPANY USING LSTM.

# ## ABOUT DATASET
# ### Google Stock Prediction
# 
# This dataset contains historical data of **Google's stock prices** and related attributes. It consists of 14 columns and a smaller subset of 1257 rows. Each column represents a specific attribute, and each row contains the corresponding values for that attribute.
# 
# The columns in the dataset are as follows:
# 
# 1. **Symbol**: The name of the company, which is **GOOG** in this case.
# 2. **Date**: The year and date of the stock data.
# 3. **Close**: The closing price of Google's stock on a particular day.
# 4. **High**: The highest value reached by Google's stock on the given day.
# 5. **Low**: The lowest value reached by Google's stock on the given day.
# 6. **Open**: The opening value of Google's stock on the given day.
# 7. **Volume**: The trading volume of Google's stock on the given day, i.e., the number of shares traded.
# 8. **adjClose**: The adjusted closing price of Google's stock, considering factors such as dividends and stock splits.
# 9. **adjHigh**: The adjusted highest value reached by Google's stock on the given day.
# 10. **adjLow**: The adjusted lowest value reached by Google's stock on the given day.
# 11. **adjOpen**: The adjusted opening value of Google's stock on the given day.
# 12. **adjVolume**: The adjusted trading volume of Google's stock on the given day, accounting for factors such as stock splits.
# 13. **divCash**: The amount of cash dividend paid out to shareholders on the given day.
# 14. **splitFactor**: The split factor, if any, applied to Google's stock on the given day. A split factor of 1 indicates no split.
# 
# The dataset is available at Kaggle : https://www.kaggle.com/datasets/shreenidhihipparagi/google-stock-prediction

# ## STEPS INVOLVED : 
# ### 1 . IMPORTING LIBRARIES AND DATA TO BE USED
# ### 2. GATHERING INSIGHTS
# ### 3. DATA PRE-PROCESSING
# ### 4. CREATING LSTM MODEL
# ### 5. VISUALIZING ACTUAL VS PREDICTED DATA
# ### 6. PREDICTING UPCOMING 15 DAYS

# 

# ## STEP 1 : IMPORTING LIBRARIES AND DATA TO BE USED

# In[42]:


#importing libraries to be used
import numpy as np # for linear algebra
import pandas as pd # data preprocessing
import matplotlib.pyplot as plt # data visualization library
import seaborn as sns # data visualization library
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore') # ignore warnings 

from sklearn.preprocessing import MinMaxScaler # for normalization
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional


# In[43]:


df = pd.read_csv('Google Stocks.csv') # data_importing
df.head(10) # fetching first 10 rows of dataset


# ## STEP 2 : GATHERING INSIGHTS

# In[44]:


# shape of data
print("Shape of data:",df.shape)


# In[45]:


# statistical description of data
df.describe()


# In[46]:


# summary of data
df.info()


# In[47]:


# checking null values
df.isnull().sum()


# ### There are no null values in the dataset

# In[48]:


df = df[['date','open','close']] # Extracting required columns
df['date'] = pd.to_datetime(df['date'].apply(lambda x: x.split()[0])) # converting object dtype of date column to datetime dtype
df.set_index('date',drop=True,inplace=True) # Setting date column as index
df.head(10)


# In[49]:


# plotting open and closing price on date index
fig, ax =plt.subplots(1,2,figsize=(20,7))
ax[0].plot(df['open'],label='Open',color='green')
ax[0].set_xlabel('Date',size=15)
ax[0].set_ylabel('Price',size=15)
ax[0].legend()

ax[1].plot(df['close'],label='Close',color='red')
ax[1].set_xlabel('Date',size=15)
ax[1].set_ylabel('Price',size=15)
ax[1].legend()

fig.show()


# ## STEP 3 : DATA PRE-PROCESSING

# In[50]:


# normalizing all the values of all columns using MinMaxScaler
MMS = MinMaxScaler()
df[df.columns] = MMS.fit_transform(df)
df.head(10)


# In[51]:


# splitting the data into training and test set
training_size = round(len(df) * 0.75) # Selecting 75 % for training and 25 % for testing
training_size


# In[52]:


train_data = df[:training_size]
test_data  = df[training_size:]

train_data.shape, test_data.shape


# In[53]:


# Function to create sequence of data for training and testing

def create_sequence(dataset):
  sequences = []
  labels = []

  start_idx = 0

  for stop_idx in range(50,len(dataset)): # Selecting 50 rows at a time
    sequences.append(dataset.iloc[start_idx:stop_idx])
    labels.append(dataset.iloc[stop_idx])
    start_idx += 1
  return (np.array(sequences),np.array(labels))


# In[54]:


train_seq, train_label = create_sequence(train_data) 
test_seq, test_label = create_sequence(test_data)
train_seq.shape, train_label.shape, test_seq.shape, test_label.shape


# ## STEP 4 :  CREATING LSTM MODEL

# In[55]:


# imported Sequential from keras.models 
model = Sequential()
# importing Dense, Dropout, LSTM, Bidirectional from keras.layers 
model.add(LSTM(units=50, return_sequences=True, input_shape = (train_seq.shape[1], train_seq.shape[2])))

model.add(Dropout(0.1)) 
model.add(LSTM(units=50))

model.add(Dense(2))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

model.summary()


# In[56]:


# fitting the model by iterating the dataset over 100 times(100 epochs)
model.fit(train_seq, train_label, epochs=100,validation_data=(test_seq, test_label), verbose=1)


# In[57]:


# predicting the values after running the model
test_predicted = model.predict(test_seq)
test_predicted[:5]


# In[58]:


# Inversing normalization/scaling on predicted data 
test_inverse_predicted = MMS.inverse_transform(test_predicted)
test_inverse_predicted[:5]


# ## STEP 5 :  VISUALIZING ACTUAL VS PREDICTED DATA

# In[59]:


# Merging actual and predicted data for better visualization
df_merge = pd.concat([df.iloc[-264:].copy(),
                          pd.DataFrame(test_inverse_predicted,columns=['open_predicted','close_predicted'],
                                       index=df.iloc[-264:].index)], axis=1)


# In[60]:


# Inversing normalization/scaling 
df_merge[['open','close']] = MMS.inverse_transform(df_merge[['open','close']])
df_merge.head()


# In[61]:


# plotting the actual open and predicted open prices on date index
df_merge[['open','open_predicted']].plot(figsize=(10,6))
plt.xticks(rotation=45)
plt.xlabel('Date',size=15)
plt.ylabel('Stock Price',size=15)
plt.title('Actual vs Predicted for open price',size=15)
plt.show()


# In[62]:


# plotting the actual close and predicted close prices on date index 
df_merge[['close','close_predicted']].plot(figsize=(10,6))
plt.xticks(rotation=45)
plt.xlabel('Date',size=15)
plt.ylabel('Stock Price',size=15)
plt.title('Actual vs Predicted for close price',size=15)
plt.show()


# ## STEP 6. PREDICTING UPCOMING 10 DAYS

# In[63]:


# Creating a dataframe and adding 10 days to existing index 

df_merge = df_merge.append(pd.DataFrame(columns=df_merge.columns,
                                        index=pd.date_range(start=df_merge.index[-1], periods=11, freq='D', closed='right')))
df_merge['2021-06-09':'2021-06-16']


# In[64]:


# creating a DataFrame and filling values of open and close column
upcoming_prediction = pd.DataFrame(columns=['open','close'],index=df_merge.index)
upcoming_prediction.index=pd.to_datetime(upcoming_prediction.index)


# In[65]:


curr_seq = test_seq[-1:]

for i in range(-10,0):
  up_pred = model.predict(curr_seq)
  upcoming_prediction.iloc[i] = up_pred
  curr_seq = np.append(curr_seq[0][1:],up_pred,axis=0)
  curr_seq = curr_seq.reshape(test_seq[-1:].shape)


# In[66]:


# inversing Normalization/scaling
upcoming_prediction[['open','close']] = MMS.inverse_transform(upcoming_prediction[['open','close']])


# In[67]:


# plotting Upcoming Open price on date index
fig,ax=plt.subplots(figsize=(10,5))
ax.plot(df_merge.loc['2021-04-01':,'open'],label='Current Open Price')
ax.plot(upcoming_prediction.loc['2021-04-01':,'open'],label='Upcoming Open Price')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
ax.set_xlabel('Date',size=15)
ax.set_ylabel('Stock Price',size=15)
ax.set_title('Upcoming Open price prediction',size=15)
ax.legend()
fig.show()


# In[68]:


# plotting Upcoming Close price on date index
fig,ax=plt.subplots(figsize=(10,5))
ax.plot(df_merge.loc['2021-04-01':,'close'],label='Current close Price')
ax.plot(upcoming_prediction.loc['2021-04-01':,'close'],label='Upcoming close Price')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
ax.set_xlabel('Date',size=15)
ax.set_ylabel('Stock Price',size=15)
ax.set_title('Upcoming close price prediction',size=15)
ax.legend()
fig.show()


# # THANK YOU!

# In[ ]:




