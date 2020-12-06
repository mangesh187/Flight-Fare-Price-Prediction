#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)


# In[2]:


train = pd.read_csv('flight_price_train.csv')
test = pd.read_csv('flight_price_test.csv')


# In[3]:


train.info()


# In[4]:


train.head()


# In[5]:


train.isnull().sum()


# In[6]:


test.describe()


# In[7]:


test.info()


# In[8]:


test.isnull().sum()


# In[9]:


train.fillna(value=0,
    method=None,
    axis=0,
    inplace=True)


# In[10]:


train.isnull().sum()


# In[11]:


fig = plt.figure(figsize=(25, 15))
cols = 5
rows = np.ceil(float(train.shape[1]) / cols)

for i, column in enumerate(train.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if train.dtypes[column] == np.object:
        train[column].value_counts().plot(kind="bar", axes=ax)
    else:
        train[column].hist(axes=ax)
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.2)


# In[12]:


train['Additional_Info'].unique()


# In[13]:


train['Journey_Day'] = pd.to_datetime(train.Date_of_Journey, format='%d/%m/%Y').dt.day
train['Journey_Month'] = pd.to_datetime(train.Date_of_Journey, format='%d/%m/%Y').dt.month
train['weekday']= pd.to_datetime(train.Date_of_Journey, format='%d/%m/%Y').dt.weekday
train.drop(['Date_of_Journey'], 1, inplace = True)


# In[14]:


test['Journey_Day'] = pd.to_datetime(test.Date_of_Journey, format='%d-%m-%Y').dt.day
test['Journey_Month'] = pd.to_datetime(test.Date_of_Journey, format='%d-%m-%Y').dt.month
test['weekday']= pd.to_datetime(test.Date_of_Journey, format='%d-%m-%Y').dt.weekday
test.drop(['Date_of_Journey'], 1, inplace = True)


# In[15]:


def duration(test):
    test = test.strip()
    total=test.split(' ')
    to=total[0]
    hrs=(int)(to[:-1])*60
    if((len(total))==2):
        mint=(int)(total[1][:-1])
        hrs=hrs+mint
    test=str(hrs)
    return test
train['Duration'] = train['Duration'].apply(duration)
test['Duration'] = test['Duration'].apply(duration)


# In[16]:


# from datetime import timedelta
# for h in train['Duration']:
# #     h = 
#     delta = timedelta(hours=int(h.split(':')[0]), minutes=int(h.split(':')[1]))
#     minutes = delta.total_seconds()/60
#     print(minutes)


# In[17]:


train.Duration.nunique()


# In[18]:


test.head(2)


# In[19]:


test['Arrival_Time'] = pd.to_datetime(test['Arrival_Time'])
test['Arrival_Time'] = test['Arrival_Time'].dt.strftime('%H:%M')


# In[20]:


def deparrtime(x):
    x=x.strip()
    tt=(int)(x.split(':')[0])
    if(tt>=16 and tt<21):
        x='Evening'
    elif(tt>=21 or tt<5):
        x='Night'
    elif(tt>=5 and tt<11):
        x='Morning'
    elif(tt>=11 and tt<16):
        x='Afternoon'
    return x
train['Dep_Time']=train['Dep_Time'].apply(deparrtime)
train['Arrival_Time']=train['Arrival_Time'].apply(deparrtime)
test['Dep_Time']=test['Dep_Time'].apply(deparrtime)
test['Arrival_Time']=test['Arrival_Time'].apply(deparrtime)


# In[21]:


train.head(2)


# In[22]:


test.head(2)


# In[23]:


def stops(x):
    if(x=='non-stop'):
        x=str(0)
    else:
        str(x).strip()
        stps=str(x).split(' ')[0]
        x=stps
    return x
train['Total_Stops'] = train['Total_Stops'].apply(stops)
test['Total_Stops'] = test['Total_Stops'].apply(stops)


# In[24]:


train.head(2)


# In[25]:


test.head(2)


# In[26]:


train['Additional_Info'].unique()


# In[27]:


pd.options.mode.chained_assignment = None 
for i in range(train.shape[0]):
    if(train.iloc[i]['Additional_Info'] == 'No info'):
        train.iloc[i]['Additional_Info'] = 'No Info'


# In[28]:


train = train.drop(['Route'], axis=1) #we don't need it as we already have total_stops
test = test.drop(['Route'], axis=1)


# In[29]:


train.head(3)


# In[30]:


train.info()


# In[31]:


train['Duration'] = train['Duration'].astype(int)
train['Journey_Day'] = train['Journey_Day'].astype(object)
train['Journey_Month'] = train['Journey_Month'].astype(object)
train['weekday'] = train['weekday'].astype(object)


# In[32]:


test['Duration'] = test['Duration'].astype(int)
test['Journey_Day'] = test['Journey_Day'].astype(object)
test['Journey_Month'] = test['Journey_Month'].astype(object)
test['weekday'] = test['weekday'].astype(object)


# In[33]:


train.head(2)


# In[34]:


train.describe()


# In[35]:


# train = train[train['Price'] < 13000]


# In[36]:


train['Journey_Month'] = train['Journey_Month'].replace({3:'March', 4:'April', 5:'May', 6:'June'})
test['Journey_Month'] = test['Journey_Month'].replace({3:'March', 4:'April', 5:'May', 6:'June'})


# In[37]:


train['Journey_Month'] = train['Journey_Month'].astype(object)
test['Journey_Month'] = test['Journey_Month'].astype(object)
train.head(2)


# In[38]:


print(train.shape)
print(test.shape)


# In[39]:


test.head(2)


# In[40]:


train.info()


# In[ ]:





# In[41]:


df = train.copy()


# ### EDA

# In[42]:


#duration v/s AveragePrice
plt.figure(figsize=(15,12))
sns.scatterplot(data=train, x='Duration', y='Price',ci=0.01, hue ='Destination',alpha=1,)


# ### Analysis :
# * We know that duration (ie.distance) plays a major role in flight ticket prices but we see no such pattern here,
# as there must be there are other significant factors affecting flight ticket price like type of airline,
# destination of flight,date of journey of flight and higher if collides with a public holiday .

# In[43]:


# Journey month vs Price
v1 = sns.barplot(x = 'Journey_Month', y='Price', data=df , estimator=sum,ci=95,
    n_boot=50,
    units=None,
    seed=None,
    orient=None,
    color=None,
    palette=None,
    saturation=0.95,
    errcolor='.66',
    errwidth=None,
    capsize=0,
    dodge=True,
    ax=None,
  )
v1.set_title('Month_Vs_Price')
v1.set_ylabel('Price')
v1.set_xlabel('Month')
v1.set_xticklabels(v1.get_xticklabels(), rotation=45)


# In[44]:


#count of flights per month
Top_months = df['Journey_Month'].value_counts().head(10)
Top_months


# ### Analysis :
# * We see that total count of flight is maximum towards the month-May which can also be concluded from the above bar plot which shows that the sum of fare is maximum in May.
# * This can be due to : Summer vacations in the month of may for schools and colleges, hence most families are also generally going for vacations around this time.
# * The count of flights is lowest on the month of April, this can be because : Schools and colleges have their final exams around this time, offices are mostly busy in the month of April.

# In[45]:


# Count of flights with different Airlines
plt.figure(figsize = (15, 10))
plt.title('Count of flights with different Airlines')
ax=sns.countplot(x = 'Airline', data =train)
ax.set_xticklabels(ax.get_xticklabels(), rotation='verticle')
plt.xlabel('Airline')
plt.ylabel('Count of flights')
plt.xticks(rotation = 90)
for i in ax.patches:
    ax.annotate(int(i.get_height()), (i.get_x()+0.25, i.get_height()+1), va='bottom',
                    color= 'White')


# In[46]:


p = df['Airline'].value_counts()
plt.figure(figsize=(12,12))
plt.pie(p.values, labels=p.index, autopct='%1.1f%%')


# ### Analysis :
# * from the chart we can see that most of people prefer Airline_A as compared to others Airlines.

# In[47]:


# Airline vs AveragePrice
sns.catplot(y = 'Price',x = 'Airline',data= train.sort_values('Price',ascending=False),kind="boxen",height=9, aspect=2)
plt.title('Airline Vs Price')
plt.show


# ### Analysis :
# *  From graph we can see that  Airline_J have the highest flight ticket Price as compared to other airlines
# and Airline_L have the lowest flight ticket price as compared to other airlines.

# In[48]:


# Source vs Price
sns.catplot(y = "Price", x = "Source", data = df.sort_values("Price", ascending = False), kind="boxen", height = 9, aspect = 2)
plt.title('Source Vs Price')
plt.show()


# ### Analysis :
# * From graph we can see that Banglore have the highest flight ticket price as compared to other sources and Chennai have the lowest flight ticket price as compared to other airlines.

# In[49]:


sns.catplot(y = "Price", x = "Destination", data = df.sort_values("Price", ascending = False), kind="boxen", height = 9, aspect = 2)
plt.title('Destination Vs Price')
plt.show()


# ### Analysis :
# * From Destination vs Price graph we can see that New Delhi have the highest flight ticket Price as compared to other Destinations.

# In[50]:


#Deptarure time v/s Price
v2 = sns.barplot(x='Dep_Time', y='Price', data = df)
v2.set_ylabel('Price')
v2.set_xlabel('Time of Departure')
v2.set_xticklabels(v2.get_xticklabels(), rotation=45)


# In[51]:


# time of departure v/s count of flights
Top_most_Departure_time = df['Dep_Time'].value_counts().head()
Top_most_Departure_time


# ### Analysis:
# * Early Morning flights are always cheaper as compare to night flight ticket prices.
# * Evening flight ticket prices are expensive due to more demand and is the most convenient time to tarvel for most people.

# In[52]:


#Arrival time v/s Price
v3 = sns.barplot(x = 'Arrival_Time', y = 'Price', data = df)
v3.set_ylabel('Price')
v3.set_xlabel('Time of Arrival')
v3.set_xticklabels(v2.get_xticklabels(), rotation=45)


# In[53]:


# Top_most_Arrival_time = df['Arrival_Time'].value_counts().head()
# Top_most_Arrival_time


# In[54]:


v4 = sns.barplot(x = 'Total_Stops', y = 'Price', data = df)
v4.set_ylabel('Price')
v4.set_xlabel('Total_Stops')
v4.set_xticklabels(v2.get_xticklabels(), rotation=45)


# ### Analysis :
# * From graph we can see that in the Afternoon Price is high (because of more stops) as compared to others.

# In[55]:


#Journey_Day v/s Average price
v5 = sns.barplot(x='Journey_Day', y='Price', data=train)
v5.set_title('Price of flights with different datess')
v5.set_ylabel('Price')
v5.set_xlabel('date')
v5.set_xticklabels(v5.get_xticklabels(), rotation=45)


# ### Analysis :
# * It looks like that there's a trend in the air fare when compared to the day of respective months, prices are higher in the start of month but this is not a trend if you see from the broader perspective as this might be due to various reasons. For eg. the date of Journey is 12th March and people are booking towards 8th March or so, this will lead to higher flight prices.(Prices increase as near you date of booking is to the date of journey). So flight prices don't follow any particular pattern towards any time of the month.

# In[56]:


df.describe()


# In[57]:


df.info()


# In[58]:


train.columns


# In[59]:


categorical_data = train.select_dtypes('object')
categorical_data = categorical_data.drop(['Additional_Info','Destination'],1)

test_categorical_data = test.select_dtypes('object')
test_categorical_data = test_categorical_data.drop(['Additional_Info','Destination'],1)

numerical_data = train.select_dtypes('int','float')
test_numerical_data = test.select_dtypes('int','float')


# In[60]:


print(categorical_data.shape)
print(test_categorical_data.shape)


# In[61]:


categorical_data


# In[62]:


test_numerical_data.head()


# In[63]:


numerical_data.head()


# In[64]:


# df_t = pd.get_dummies(df,columns=['Airline', 'Source', 'Destination', 'Dep_Time', 'Arrival_Time',
#         'Total_Stops', 'Additional_Info', 'Journey_Day', 'Journey_Month', 'weekday'],drop_first = False)
# df_test = pd.get_dummies(df,columns=['Airline', 'Source', 'Destination', 'Dep_Time', 'Arrival_Time',
#         'Total_Stops', 'Additional_Info', 'Journey_Day', 'Journey_Month', 'weekday'],drop_first = False)


# In[65]:


#Label encode and hot encode categorical columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_categorical_data = categorical_data.apply(le.fit_transform)
test_categorical_data = test_categorical_data.apply(le.fit_transform)
train_categorical_data.head()


# In[66]:


test_categorical_data.head()


# In[ ]:





# In[67]:


print(train_categorical_data.shape)
print(test_categorical_data.shape)


# In[68]:


df_train = pd.concat([train_categorical_data, numerical_data],axis=1)
df_test = pd.concat([test_categorical_data, test_numerical_data],axis=1)

df_train.info()


# In[69]:


df_train.describe()


# In[70]:


df_train1 = df_train[df_train['Price'] < 12374]
df_train1


# In[71]:


print(df_train.shape)
print(df_test.shape)


# In[72]:


X = df_train.drop(['Price'],1)#df_train.drop(['Price'],1)
y = df_train['Price']
from sklearn.ensemble import ExtraTreesRegressor

model_features_importance=ExtraTreesRegressor()
model_features_importance.fit(X,y)
print(model_features_importance.feature_importances_)
ranked_features = pd.Series(model_features_importance.feature_importances_,index=X.columns)


# In[73]:


ranked_features.nlargest(7).plot(kind='barh')
plt.show()


# In[74]:


top_features = ranked_features.nlargest(7).index
train_df = df_train[top_features]
test_df = df_test[top_features]


# In[75]:


print(train_df.shape)
print(test_df.shape)


# In[76]:


# train_df.columns == test_df.columns


# In[77]:


# plt.figure(figsize=(15,12))
# sns.heatmap(f_df.corr(),annot=True, fmt='.1g', cmap="BrBG")


# In[78]:


X = train_df#.drop(['Price'],1)
y = df_train['Price']


# In[79]:


# training testing and splitting the dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor


# In[80]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
model = RandomForestRegressor( n_estimators=125,
                                criterion='mse',
                                max_depth=60,
                                min_samples_split=2,
                                min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0,
                                max_features=7,
                                max_leaf_nodes=None,
                                min_impurity_decrease=0.0,
                                min_impurity_split=None,
                                bootstrap=True,
                                oob_score=False,
                                n_jobs=None,
                                random_state=42,
                                verbose=0,
                                warm_start=False,
                                ccp_alpha=0.0,
                                max_samples=None,
                             )
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('RMSE:', np.sqrt(mse(y_test, y_pred)))
rmse = -np.sqrt(np.square(np.log10(y_pred +1) - np.log10(y_test +1)).mean())
print('rmse:', rmse)


# In[81]:


X_train.shape


# In[82]:


model.fit(X, y)


# In[83]:


# print("The size of training input is", X_train.shape)
# print("The size of training output is", y_train.shape)
# print(40 *'*')
# print("The size of testing input is", X_test.shape)
# print("The size of testing output is", y_test.shape)


# In[84]:


from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

tuned_params = {'max_depth': [1,3,5,10,25,50,80,250], 
                'learning_rate': [0.001, 0.01, 0.05],
                'n_estimators': [100, 150, 250],
                'reg_lambda': [ 0.1, 1.0, 10.0]
               }
model = RandomizedSearchCV(XGBRegressor(), 
                           tuned_params, 
                           n_iter=20,
                           scoring = 'neg_root_mean_squared_error',
                           cv = 15, n_jobs=-1
                          )


# In[85]:

# In[86]:


model.fit(X,y)


# In[87]:


y_pred = model.predict(test_df)
# print('RMSE : ',np.sqrt(mse(y_test, y_pred)))


# In[89]:


-np.sqrt(np.square(np.log10(y_pred +1) - np.log10(y_test +1)).mean())


# In[90]:


# y_test_pred = model.predict(df_test)


# In[91]:


output = pd.DataFrame(data={"Price":y_pred})

output.shape

output.to_csv('Mangesh_flight_f.csv',index=False)


# In[92]:


# In[ ]:
