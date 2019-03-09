import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14
plt.style.use("fivethirtyeight")
url = './data/bikeshare.csv'
bikes = pd.read_csv(url, index_col='datetime', parse_dates=True)

bikes['temp_F'] = bikes.temp*1.8 + 32

print('test')
print(bikes.head())
bikes.rename(columns={'count':'total_rentals'}, inplace=True)
feature_cols = ['temp']
X = bikes[feature_cols]
y = bikes.total_rentals

X2= bikes[['temp_F']]
print((type(X)))
print((type(X.values)))
print((type(y)))
print((type(y.values)))
print((X.shape))
print((y.shape))

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
type(lr)

lr.fit(X,y)

linreg = LinearRegression()
linreg.fit(X2, y)
print(lr.intercept_)
print(lr.coef_)
print(linreg.intercept_)
print(linreg.coef_)

print(lr.predict([[0],[10]]))
print(linreg.predict([[32],[50]]))