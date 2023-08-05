### Tunnel Traffic
from pathlib import Path
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


simplefilter('ignore')

## 
# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 4))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)

import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('retina')

##Load traffic data
data_dir = Path('./input/ts-course-data')
tunnel = pd.read_csv(data_dir / "tunnel.csv", parse_dates=["Day"])

# Create a time series in Pandas by setting the index to a date
# column. We parsed "Day" as a date type by using `parse_dates` when
# loading the data.
tunnel = tunnel.set_index('Day')


# By default, Pandas creates a `DatetimeIndex` with dtype `Timestamp`
# (equivalent to `np.datetime64`, representing a time series as a
# sequence of measurements taken at single moments. A `PeriodIndex`,
# on the other hand, represents a time series as a sequence of
# quantities accumulated over periods of time. Periods are often
# easier to work with, so that's what we'll use in this course.

tunnel = tunnel.to_period()

print(tunnel.head())


## Time Step
df = tunnel.copy()
df['Time'] = np.arange(len(df.index))

print(df.head())

## This can be done in scikit-learn
'''
The procedure for fitting a linear regression model follows the standard steps for scikit-learn.

'''
from sklearn.linear_model import LinearRegression

#Training Data
X = df.loc[:, ['Time']] #features
y = df.loc[:, 'NumVehicles'] #Target

#Train the model
model = LinearRegression()
model.fit(X,y)

#Pred
y_pred = pd.Series(model.predict(X),index=X.index)

ax = y.plot(**plot_params)
ax = y_pred.plot(ax=ax, linewidth=3)
ax.set_title('Time Plot of Tunnel Traffic');

plt.show()

## LAG Feature
df['Lag_1'] = df['NumVehicles'].shift(1)
print(df.head())

'''
When creating lag features, we need to decide what to do with the missing values produced. 
Filling them in is one option, maybe with 0.0 or "backfilling" with the first known value. 
Instead, we'll just drop the missing values, making sure to also drop values in the target from corresponding dates.
'''

X = df.loc[:, ['Lag_1']]
X.dropna(inplace=True) # drop missing values in the feature set
y = df.loc[:, 'NumVehicles']  # create the target

y,X = y.align(X, join='inner') #drop the corresponding values in target

## Model
model = LinearRegression()
model.fit(X, y)



#Pred
y_pred = pd.Series(model.predict(X), index=X.index)

fig, ax = plt.subplots()
ax.plot(X['Lag_1'], y, '.', color='0.25')
ax.plot(X['Lag_1'], y_pred)
ax.set_aspect('equal')
ax.set_ylabel('NumVehicles')
ax.set_xlabel('Lag_1')
ax.set_title('Lag Plot of Tunnel Traffic');
plt.show()


ax = y.plot(**plot_params)
ax = y_pred.plot()
plt.show()

