import math
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#----------------
#| Question 1   |
#----------------
datasets = sns.get_dataset_names()
print(f"Question 1 Answer:\n")

for data in datasets:
    print(data)

#----------------
#| Question 2   |
#----------------

# Load datasets
diamonds = sns.load_dataset("diamonds")
iris = sns.load_dataset("iris")
tips = sns.load_dataset("tips")
penguins = sns.load_dataset("penguins")
titanic = sns.load_dataset("titanic")

# Store # of observations
diamonds_num_observations = diamonds.shape[0]
iris_num_observations = iris.shape[0]
tips_num_observations = tips.shape[0]
penguins_num_observations = penguins.shape[0]
titanic_num_observations = titanic.shape[0]


# Get list of categorical features
# The simple select_dtypes() function worked for most of the datasets
# Additional logic was needed for the titanic dataset,
# which contains numeric columns that can be classified as categorical
diamonds_categorical_features = diamonds.select_dtypes(include=['category', 'object']).columns.tolist()
iris_categorical_features = iris.select_dtypes(include=['category', 'object']).columns.tolist()
tips_categorical_features = tips.select_dtypes(include=['category', 'object']).columns.tolist()
penguins_categorical_features = penguins.select_dtypes(include=['category', 'object']).columns.tolist()
titanic_categorical_features = titanic.select_dtypes(include=['category', 'object', 'boolean']).columns.tolist()

# Grab all binary columns that use 0,1 and add them to the categorical features
titanic_binary_categorical_features = titanic.columns[titanic.isin([0,1]).all()]
for name in titanic_binary_categorical_features:
    if name not in titanic_categorical_features:
        titanic_categorical_features.append(name)

# pclass is a numeric ordinal column, so I am going to add it to the categorical features
titanic_categorical_features.append('pclass')

# Get list of numerical features
# Once again, the titanic dataset requires additional logic to
# handle numeric columns that are categorical
diamonds_numerical_features = diamonds._get_numeric_data().columns.tolist()
iris_numerical_features = iris._get_numeric_data().columns.tolist()
tips_numerical_features = tips._get_numeric_data().columns.tolist()
penguins_numerical_features = penguins._get_numeric_data().columns.tolist()
titanic_numerical_features = titanic._get_numeric_data().columns.tolist()

# Remove the binary columns from the numerical features
for name in titanic_binary_categorical_features:
    if name in titanic_numerical_features:
        titanic_numerical_features.remove(name)

# Remove pclass from the numerical features because it is ordinal,
# which is a subsection of categorical
titanic_numerical_features.remove('pclass')

# Display # of observations
print(f"\nQuestion 2:\nDiamonds # of Observations: {diamonds_num_observations}\n"
      f"Iris # of Observations: {iris_num_observations}\n"
      f"Tips # of Observations: {tips_num_observations}\n"
      f"Penguins # of Observations: {penguins_num_observations}\n"
      f"Titanic # of Observations: {titanic_num_observations}\n")

# Display Numerical Features
print(f"Question 2:\nDiamonds Numerical Features:\n{diamonds_numerical_features}\n\n"
      f"Iris Numerical Features:\n{iris_numerical_features}\n\n"
      f"Tips Numerical Features:\n{tips_numerical_features}\n\n"
      f"Penguins Numerical Features:\n{penguins_numerical_features}\n\n"
      f"Titanic Numerical Features:\n{titanic_numerical_features}\n")

# Display Categorical Features
print(f"Question 2:\nDiamonds Categorical Features:\n{diamonds_categorical_features}\n\n"
      f"Iris Categorical Features:\n{iris_categorical_features}\n\n"
      f"Tips Categorical Features:\n{tips_categorical_features}\n\n"
      f"Penguins Categorical Features:\n{penguins_categorical_features}\n\n"
      f"Titanic Categorical Features:\n{titanic_categorical_features}\n")
#----------------
#| Question 3   |
#----------------

# Load dataset
titanic = sns.load_dataset("titanic")

# Display the count, mean, std, min, 25%, 50%, 75% and max for the numerical features in the dataset

# Grab only the numerical features from the dataset
numerical_features = titanic[['age', 'sibsp', 'parch', 'fare']]

print(f"\nQuestion 3:\nCount, mean, std, min, 25%, 50%, 75% and max:\n{numerical_features.describe()}")

# Display # of missing observations
print(f"\nQuestion 3:\nNumber of Missing Observations:\n{titanic.isnull().sum()}")

#----------------
#| Question 4   |
#----------------

titanic = sns.load_dataset('titanic')

# Grab only numerical features and store to new variable
titanic_numerical_features = titanic.drop(columns=['survived',
                                                   'pclass',
                                                   'sex',
                                                   'embarked',
                                                   'class',
                                                   'who',
                                                   'adult_male',
                                                   'deck',
                                                   'embark_town',
                                                   'alive',
                                                   'alone'])

# Display head of old dataset and new one
pd.set_option('display.max_columns', None)
print(f"\nQuestion 4:\nOriginal head:\n{titanic.head()}"
      f"\nNew head:\n{titanic_numerical_features.head()}")

#----------------
#| Question 5   |
#----------------

# Record shape before dropping any observations
old_shape = titanic_numerical_features.shape

# Drop any observation/row with missing attributes
titanic_numerical_features.dropna(inplace=True)

# Calculate number of deleted rows
del_rows = old_shape[0] - titanic_numerical_features.shape[0]

# Calculate the percentage of deleted data
percent_del_data = (del_rows / old_shape[0]) * 100

# Display number of deleted observations
print(f"\nQuestion 5:\nNumber of deleted observations:\n{del_rows}"
      f"\nPercentage of deleted data:\n{round(percent_del_data, 2)}%")

#----------------
#| Question 8   |
#----------------

# Geometric mean function
def geometric_mean(data):
    # Check for zeroes or negative numbers
    if any(value <= 0 for value in data):
        return 0

    # Using log to avoid overflow
    log_values = np.log(series)
    mean_log = np.mean(log_values)
    return np.exp(mean_log)

# Harmonic mean function
def harmonic_mean(data):
    # Check for zeroes and negative numbers
    if any(value <= 0 for value in data):
        return 0

    # Convert series to a numpy array for efficient operations
    series = np.array(data)

    # Calculate the harmonic mean
    return len(series) / np.sum(1.0 / series)

# Store geometric mean results
geometric_results = {}

# Grab data from each column and pass it to geometric_mean function
for column in titanic_numerical_features.columns:
    series = titanic_numerical_features[column]
    geometric_results[column] = geometric_mean(series)

# Store harmonic mean results
harmonic_results = {}

# Grab data from each column and pass it to harmonic_mean function
for column in titanic_numerical_features.columns:
    series = titanic_numerical_features[column]
    harmonic_results[column] = harmonic_mean(series)

# Display arithmetic mean
print(f"\nQuestion 8:\nArithmetic Mean:\n{titanic_numerical_features.mean()}\n")

# Display geometric mean
print(f"Geometric Mean:")
for col in geometric_results:
    print(col, ' : ', geometric_results[col])

# Display harmonic mean
print(f"\nHarmonic Mean:")
for col in harmonic_results:
    print(col, ' : ', harmonic_results[col])

#----------------
#| Question 9   |
#----------------

# Focus on just the age column/feature
age = titanic_numerical_features.drop(columns=['sibsp',
                                                'parch',
                                                'fare'])
# Make histplot of age data
sns.histplot(data=age)
plt.title("Question 9: Histogram For Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Focus on just the fare column/feature
fare = titanic_numerical_features.drop(columns=['sibsp',
                                                'parch',
                                                'age'])

# Make histplot of fare data
sns.histplot(data=fare)
plt.title("Question 9: Histogram For Fare")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.show()

#----------------
#| Question 10  |
#----------------

# Plot and display the pairwise bivariate distribution
pd.plotting.scatter_matrix(titanic_numerical_features, alpha=0.2)
plt.tight_layout()
plt.show()