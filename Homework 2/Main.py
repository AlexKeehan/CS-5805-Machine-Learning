import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from fontTools.merge.util import first

# Question 1

# Read in file
stock_prices = pd.read_csv("stock prices.csv")

# Display all features & their missing values
print(f"Question 1:\nNull Values Before Cleaning:\n{stock_prices.isnull().sum()}")

# Display the total # of missing values in the entire dataset
print(f"\nTotal # Of Null Values Before Cleaning: {stock_prices.isnull().values.sum()}")


# Grab columns that contain null values and replace them with the mean of the column
for col in stock_prices.columns:
    if stock_prices[col].isnull().sum() != 0:
        mean = stock_prices[col].mean()
        stock_prices.fillna(mean, inplace=True)

# Display features after cleaning
print(f"\nQuestion 1:\nNull Values After Cleaning:\n{stock_prices.isnull().sum()}")

# Display total # of missing values after cleaning
print(f"\nTotal # Of Null Values After Cleaning: {stock_prices.isnull().values.sum()}")

# Question 2

# Remove duplicate companies
unique_companies = stock_prices.drop_duplicates('symbol', keep='first', inplace=False)

# Display total # of unique companies
print(f"\nQuestion 2:\nTotal # Of Unique Companies: {unique_companies.shape[0]}")

# Display the list of unique companies
print(f"\nQuestion 2:\n List of Unique Companies:\n{unique_companies['symbol']}")

# Store only Google & Apple stock in a new dataframe
tech_companies = stock_prices[stock_prices['symbol'].isin(['GOOGL', 'AAPL'])]

# Change the date col to datetime
tech_companies.loc[:, 'date'] = pd.to_datetime(tech_companies['date'])

# Pivot the df to graph it properly
pivot_df = tech_companies.pivot(index='date', columns='symbol', values='close')

# Plot it
pivot_df.plot(figsize=(12,8), use_index=True)

# Labels
plt.xlabel("Date")
plt.ylabel("USD($)")

# Use date_range to separate the date ticks out & specify how many I want
num_ticks = 6

tick_dates = pd.date_range(start=pivot_df.index.min(),
                           end=pivot_df.index.max(),
                           periods=num_ticks)

# Set x-ticks
plt.xticks(tick_dates, rotation=0, ha='center')

# Show plot
plt.title("Apple and Google Stock closing value comparison")
plt.grid()
plt.show()

# Question 3

# Group by 'symbol' & sum all the numeric cols
aggregate_stock_prices = stock_prices.groupby('symbol', as_index=False).sum(numeric_only=True)

# Set the date col as the index
aggregate_stock_prices.set_index(unique_companies['date'], inplace=True)

# Display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Display shape of old dateset & new aggregated one
print(f"\nQuestion 3:\nOld Dataset Shape: {stock_prices.shape}\n"
      f"Aggregated Dataset Shape: {aggregate_stock_prices.shape}")

# Display first 5 rows of the aggregated dataset
print(f"\nQuestion 3:\nAggregated Dataset First Five Rows:\n{aggregate_stock_prices[:5]}")

# Question 4

# Slice the dataset
sliced_stock_prices = stock_prices[['symbol', 'close', 'volume']]

# Group by symbol and aggregate with variance & mean
# Creating different cols to represent the variance & mean operations
variance_mean = sliced_stock_prices.groupby('symbol').agg(
    mean_close=('close', 'mean'),
    variance_close=('close', 'var'),
    mean_volume=('volume', 'mean'),
    variance_volume=('volume', 'var')).reset_index()

# Display dataset with aggregation operations done on it
print(f"\nQuestion 4:\nSliced Dataset With Variance & Mean Aggregation:\n{variance_mean}")

# Find max variance value for col close
max_variance = np.max(variance_mean['variance_close'])

# Find the index for that max variance
max_variance_index = np.argmax(variance_mean['variance_close'])

# Get the corresponding symbol for max variance
max_variance_symbol = variance_mean['symbol'].iloc[max_variance_index]

print(f"\nQuestion 4:\nCompany With Maximum Variance:\n"
      f"Company: {max_variance_symbol}\n"
      f"Variance: {max_variance}\n")

# Question 5

# Filter data to grab only Google data after the given date
recent_google_data = stock_prices[(stock_prices['date'] > '2015-01-01') & (stock_prices['symbol'] == 'GOOGL')]

# Filter to grab only close col
google_recent_closing_costs = recent_google_data[['date','close']]

# Display the first 5 rows of Google close col after 2015-01-01
print(f"\nQuestion 5:\nFirst Five Rows Of Google Closing Stock After Jan 1 2015:\n{google_recent_closing_costs[:5]}")

# Question 6

# Change date col to datetime
google_recent_closing_costs.loc[:, 'date'] = pd.to_datetime(google_recent_closing_costs['date'])

# Pivot data
pivoted_data = google_recent_closing_costs.pivot_table(index='date', values='close', aggfunc='mean')

# Calculate rolling mean
pivoted_data['rolling_mean'] = pivoted_data['close'].rolling(window=30, center=True).mean()

# set figure size
plt.figure(figsize=(12,8))

# Plot close data
plt.plot(pivoted_data.index,
         pivoted_data['close'],
         label="Closing Data")

# Plot rolling mean
plt.plot(pivoted_data.index,
         pivoted_data['rolling_mean'],
         label="Rolling Mean")


# Use date_range to separate the date ticks out & specify how many I want
num_ticks = 8

tick_dates = pd.date_range(start=pivoted_data.index.min(),
                           end=pivoted_data.index.max(),
                           periods=num_ticks)

# Set x-ticks
plt.xticks(tick_dates, rotation=0, ha='center')

# Labels
plt.xlabel("Date")
plt.ylabel("USD($)")

# Title & Show plot
plt.title("Google closing stock price after Jan 2015 versus Rolling window")
plt.grid()
plt.legend()
plt.show()

# Display total # of observations missed using the rolling mean of 30 days
print(f"\nQuestion 6:\nTotal Number of Missed Observations With Rolling Mean: "
      f"{google_recent_closing_costs['rolling_mean'].isnull().values.sum()}")

# Question 7

