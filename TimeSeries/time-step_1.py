import pandas as pd


## Read entries for hardcover book
df = pd.read_csv(
    './input/ts-course-data/book_sales.csv',
    index_col='Date',
    parse_dates=['Date'],
).drop('Paperback', axis=1)

print(df.head())



import numpy as np

df['Time'] = np.arange(len(df.index))

print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')
plt.rc(
    "figure",
    autolayout=True,
    figsize=(11,4),
    titlesize=18,
    titleweight='bold',
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)

import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('retina')

fig, ax = plt.subplots()
ax.plot('Time', 'Hardcover', data=df, color='0.75')
ax = sns.regplot(x='Time', y='Hardcover', data=df, ci=None, scatter_kws=dict(color='0.25'))
ax.set_title("Time plot of Hardcover Sales")

plt.show()


### LAG
df['Lag-1'] = df['Hardcover'].shift(1)
df = df.reindex(columns=['Hardcover', 'Lag_1'])

print(df.head())

