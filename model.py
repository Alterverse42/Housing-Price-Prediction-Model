### Data Science Regression Project: Predicting Home Prices in Bengaluru

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor
import pickle

#### Loading Bengluru housing prices into a dataframe

df1 = pd.read_csv("Bengaluru_House_Data.csv")
print(df1.info())
print(df1.head())
print(df1.shape)
print(df1.columns)

print(df1['area_type'].unique())

print(df1['area_type'].value_counts())


##### Dropping features that are not required to build our model
df2 = df1.drop(['area_type','society','balcony','availability'],axis='columns')
print(df2.shape)


#### Handling NA values
print(df2.isnull().sum())
print(df2.shape)

df3 = df2.dropna()

print(df3.isnull().sum())
print(df3.shape)


#### Feature Engineering

#### Adding new feature(integer) for bhk (Bedrooms Hall Kitchen)
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
print(df3.bhk.unique())


##### Exploring total_sqft feature
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

print(df3[~df3['total_sqft'].apply(is_float)].head(10))

def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

df4 = df3.copy()
df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)
df4 = df4[df4.total_sqft.notnull()]

print(df4.head(2))


##### Adding new feature: price per square feet

df5 = df4.copy()
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']
print(df5.head())

df5_stats = df5['price_per_sqft'].describe()
print(df5_stats)

##Applying dimensional reduction on location by
##Tagging Location having less than 10 data points as "other" location

df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5['location'].value_counts(ascending=False)

print(location_stats)
print(len(location_stats[location_stats<=10]))

location_stats_less_than_10 = location_stats[location_stats<=10]

print(location_stats_less_than_10)
print(len(df5.location.unique()))

df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
print(len(df5.location.unique()))
print(df5.head(10))


#### Outlier Removal Using Business Logic

print(df5[df5.total_sqft/df5.bhk<300].head())
print(df5.shape)

df6 = df5[~(df5.total_sqft/df5.bhk<300)]
print(df6.shape)


#### Outlier Removal Using Standard Deviation and Mean

print(df6.price_per_sqft.describe())

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)
print(df7.shape)

### Plotting 2BHK and 3BHK Prices against Total sq. feet area for Rajaji Nagar

def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    plt.figure(figsize  = (10,8))
    plt.scatter(bhk2.total_sqft,bhk2.price,color='lightcoral',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='*', color='darkslategray',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()

plot_scatter_chart(df7,"Rajaji Nagar")
plt.show()

## Removing those 2 BHK apartments whose price_per_sqft is less than mean price_per_sqft of 1 BHK apartment

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')

df8 = remove_bhk_outliers(df7)
# df8 = df7.copy()
print(df8.shape)

plot_scatter_chart(df8,"Rajaji Nagar")
plt.show()

plt.figure(figsize  = (10,8))
plt.hist(df8.price_per_sqft,rwidth=0.8, color = "green")
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")
plt.show()

### Outlier Removal Using Bathrooms Feature

print(df8.bath.unique())

plt.hist(df8.bath,rwidth=0.8, color = "burlywood")
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")

print(df8[df8.bath>10])
print(df8[df8.bath>df8.bhk+2])

##choosing data with (no. of bathrooms) < (no. of bedrooms + 2)
df9 = df8[df8.bath<df8.bhk+2]
print(df9.shape)
print(df9.head(4))

df10 = df9.drop(['size','price_per_sqft'],axis='columns')
print(df10.head(4))


#### Use One Hot Encoding For Location

dummies = pd.get_dummies(df10.location)
print(dummies.head(3))

df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
print(df11.head())

df12 = df11.drop('location',axis='columns')
print(df12.head(2))


#### Model building

print(df12.shape)

##Preparing data for Model
X = df12.drop(['price'],axis='columns')
X.to_csv('X.csv')
print(X.head(4))
print(X.shape)

y = df12.price
print(y.head(3))

print(len(y))
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)

##Linear Regression Model
lr = LinearRegression()
lr.fit(X_train,y_train)
print(lr.score(X_train,y_train))


predict = lr.predict(X_test)
print(r2_score(y_test, predict))

#plotting model result
plt.figure(figsize=(10, 8))
plt.scatter(y_test, predict, c='green', alpha=0.5)

max_value = max(max(predict), max(y_test))
plt.plot([0, max_value], [0, max_value], color='red', linestyle='--')

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs. Actual Values')
plt.show()


### Using Random Forest Algorithm

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf.score(X_train, y_train)

predict_rf = rf.predict(X_test)
rf.score((X_test), (y_test))


## Using K Fold cross validation to measure accuracy model

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
print("Score for Linear Regression: ", cross_val_score(LinearRegression(), X, y, cv=cv))
print("Score for RandomForestRegressor: ", cross_val_score((RandomForestRegressor()), X, y, cv=cv))

#Plotting the result of model
plt.figure(figsize=(10, 8))
plt.scatter(y_test, predict_rf, c='blue', alpha=0.5)

max_value = max(max(predict_rf), max(y_test))
plt.plot([0, max_value], [0, max_value], color='red', linestyle='--')

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs. Actual Values')
plt.show()

pickle.dump(lr, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))