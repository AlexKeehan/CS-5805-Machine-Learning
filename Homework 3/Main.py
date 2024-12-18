import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import DataPreprocessing as preproc
import numpy as np
from tabulate import tabulate

# Question 1 was completed in DataPreprocessing.py

# Question 2

# Read dataset in
df = yf.download('AAPL', start='2000-01-01', end='2022-09-25')

# Create DataPreProcessing object
preproc = preproc.DataPreProcessing()

# Plot original data
preproc.show_original(df)

# Plot normalized data
preproc.show_normalized(df)

# Plot standardized data
preproc.show_standardized(df)

# Plot IQR data
preproc.show_iqr(df)

# Question 3

# Break this logic off into a function for ease of use
def minkowski_distance(x, y, r):
    # Formula for Minkowski Distance
    return (np.abs(x)**r + np.abs(y)**r)**(1/r)

# Define the range of values for x and y to help populate the graph
x = np.linspace(-1, 1, 400)
y = np.linspace(-1, 1, 400)
X, Y = np.meshgrid(x, y)

# Define the values of r
r = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]

# Set up the figure size
plt.figure(figsize=(10, 10))

# Setup a colormap
colors = plt.cm.viridis(np.linspace(0, 1, len(r)))

# Plot each Minkowski distance
for val, color in zip(r, colors):
    distance = minkowski_distance(X, Y, val)
    # Plot the contour for each distance
    plt.contour(X, Y, distance, levels=[1], linewidths=2, colors=[color])

# Define x & y limit
plt.xlim(-1, 1)
plt.ylim(-1, 1)

# Title & labels
plt.title('Minkowski Distance Contours for Various r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid()

# Setup the legend
plt.gca().set_aspect('equal', adjustable='box')

# Create a legend with all r values & their corresponding colors
handles = [plt.Line2D([0], [0], color=color, lw=2) for color in colors]
plt.legend(handles, [f'r = {val}' for val in r], title="Values of r", loc='upper right')

# Show the plot
plt.show()

# Questions 4 & 5 were solved on paper

# Question 6

# Set random seed
np.random.seed(5808)

# Generate the random variable x
x = np.random.normal(1, np.sqrt(2), 1000)

# Generate the random variable epsilon
epsilon = np.random.normal(2, np.sqrt(3), 1000)

# Generate random variable y
y = x + epsilon


# Question 6 Part A

# Calculate covariance & variance
cov_xy = np.sum((x - np.mean(x)) * (y - np.mean(y))) / (1000 - 1)
var_x = np.sum((x - np.mean(x)) ** 2) / (1000 - 1)
var_y = np.sum((y - np.mean(y)) ** 2) / (1000 - 1)

# Construct covariance matrix
cov_matrix = np.array([[var_x, cov_xy], [cov_xy, var_y]])


# Set headers & table contents for clean display
headers = ["Estimated Covariance Matrix", "", ""]
table = [["", "x", "y"],
         ["Var(x)", var_x, cov_xy],
         ["Cov(x, y)", cov_xy, var_y]]

# Show table
print(tabulate(table, headers, tablefmt="grid", floatfmt=".2f"))

# Question 6 Part B

# Calculate the e-values & e-vectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Prepare the data for display
# Use this as the header
eigen_data = [["Eigenvalue", "Eigenvector 1", "Eigenvector 2"]]

# Append e-values into the header array accordingly
for i in range(len(eigenvalues)):
    eigen_data.append([eigenvalues[i], eigenvectors[0, i], eigenvectors[1, i]])

# Display the data in a neat table
print(tabulate(eigen_data,
               headers="firstrow",
               tablefmt="grid",
               stralign="center",
               numalign="center",
               floatfmt=".2f"))

# Question 6 Part C

# Plot x & y
plt.figure(figsize=(12, 8))
plt.scatter(x, y, alpha=0.5, color='black', label='Data points')

# Plot the eigenvectors
for i in range(len(eigenvalues)):
    # Scale the eigenvector for better visualization
    eigenvector = eigenvectors[:, i] * 2
    plt.quiver(np.mean(x), np.mean(y), eigenvector[0], eigenvector[1],
               angles='xy', scale_units='xy', scale=1,
               color=['red', 'orange'][i], label=f'Eigenvector {i + 1}')

# Set x & y limit
plt.xlim(np.min(x) - 1, np.max(x) + 1)
plt.ylim(np.min(y) - 1, np.max(y) + 1)

# Add title and axis labels
plt.title('Scatter Plot of x and y with Eigenvectors')
plt.xlabel('x-axis')
plt.ylabel('y-axis')

# Show the plot
plt.legend()
plt.grid()
plt.show()

# Question 6 Part D

X = np.vstack((x, y)).T

# Center the feature matrix
X_centered = X - np.mean(X, axis=0)

# Calculate SVDs
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

# Create header
singular_values_data = [["Singular Value"]]

# Input data into table for neat display
for value in S:
    singular_values_data.append([value])

# Display the table
print(tabulate(singular_values_data,
               headers="firstrow",
               tablefmt="grid",
               stralign="center",
               numalign="center",
               floatfmt=".2f"))

# Question 6 Part E

# Convert x & y to panda df
df = pd.DataFrame({'x': x, 'y': y})

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Grab data
correlation_data = correlation_matrix.loc['x', 'y']

# Display the table
print(tabulate([['x', 'y', correlation_data]],
               headers=['Feature 1', 'Feature 2', 'Correlation Coefficient'],
               tablefmt='grid',
               floatfmt=".2f"))

# Question 7

# Functions for differencing

# Function to perform first order differencing
def first_order_differencing(dataset):
    return np.diff(dataset)

# Function to perform second order differencing
def second_order_differencing(dataset):
    dataset = first_order_differencing(dataset)
    return np.diff(dataset)

# Function to perform third order differencing
def third_order_differencing(dataset):
    dataset = second_order_differencing(dataset)
    return np.diff(dataset)

# Create sample data
x = np.arange(-4, 5, 1)
y = x**3

# Perform differencing on y
first_diff = first_order_differencing(y)
second_diff = second_order_differencing(y)
third_diff = third_order_differencing(y)

# Create table
table_data = []

# Add data into table for neat display
for i in range(len(x)):
    table_data.append([
        x[i],
        y[i],
        first_diff[i-1] if i > 0 else None,
        second_diff[i-2] if i > 1 else None,
        third_diff[i-3] if i > 2 else None
    ])

# Define headers for table
headers = ["x(t)", "y(t)", "1st Order Differencing", "2nd Order Differencing", "3rd Order Differencing"]

# Display the table
print(tabulate(table_data, headers=headers, tablefmt='grid', floatfmt=".2f"))

# Prepare values for plotting
# One less value for first differencing
x_first_diff = x[1:]

# Two less for second differencing
x_second_diff = x[2:]

# Three less for third differencing
x_third_diff = x[3:]

# Setup figure
plt.figure(figsize=(12, 8))

# Plot
plt.plot(x, y, label='Original y(t) = x(t)^3', color='blue')
plt.plot(x_first_diff, first_diff, label='1st Order Differencing', color='orange')
plt.plot(x_second_diff, second_diff, label='2nd Order Differencing', color='green')
plt.plot(x_third_diff, third_diff, label='3rd Order Differencing', color='red')

# Adding labels and title
plt.title('Original Data and Differenced Datasets')
plt.xlabel('x(t)')
plt.ylabel('Values')
plt.grid()
plt.legend()
plt.show()