import pandas as pd


## Read entries for hardcover book
df = pd.read_csv(
    './input/ts-course-data/book_sales.csv',
    index_col='Date',
    parse_dates=['Date'],
).drop('Paperback', axis=1)

print(df.head())


### LAG
df['Lag_1'] = df['Hardcover'].shift(1)
df = df.reindex(columns=['Hardcover', 'Lag_1'])

print(df.head())

