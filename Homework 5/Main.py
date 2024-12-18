import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm
from tabulate import tabulate

# Question 1 Part A

# Read in dataset
carseat = pd.read_csv("Carseats.csv")

# Aggregate
agg_sales = carseat.groupby(['ShelveLoc', 'US'])['Sales'].sum().unstack()


# Bar plot
agg_sales.plot(kind='barh', stacked=True)
plt.title('Sales by Shelve Location and US Status')
plt.xlabel('Sales')
plt.ylabel('Shelve Location')
plt.legend(title='US Status', labels=['Outside US', 'Inside US'])
plt.grid()
plt.show()

# Question 1 Part B
carseat_encoded = pd.get_dummies(carseat, columns=["ShelveLoc",
                                                        "Urban",
                                                        "US"], drop_first=True)
# Show all cols in result
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# Print One-hot Encoding results
print(f"Question 1 Part B:\nOne-hot Encoding Results:\n{carseat_encoded.head()}\n")

# Question 1 Part C

# Function to standardize dataset
def standardized(dataset):
    # Standardize only numerical cols
    for column in dataset.select_dtypes(include=['float64', 'int64']).columns:
        dataset[column] = (dataset[column] - dataset[column].mean()) / dataset[column].std()

    return dataset

# Define features
features = carseat_encoded.drop(['Sales'], axis=1)
# Define target
target = carseat_encoded['Sales']

# Split the dataset
original_train, original_test, original_target_train, original_target_test = train_test_split(features, target, test_size=0.2, random_state=5805, shuffle=True)

# Standardize
X_train = standardized(original_train)
X_test = standardized(original_test)

# Output the head of train & test datasets
print(f"Question 1 Part C:Training Set:\n{X_train.head()}\n")
print(f"Question 1 Part C:Test Set:\n{X_test.head()}\n")

# Question 2 Part A & Part B

# Function to print the OLS summary
def print_ols_summary(model, step):
    print(f"\nOLS Summary after elimination step {step}:")
    print(model.summary())

# Function for backwards stepwise regression
def backward_stepwise_regression(X, y, threshold=0.01):

    # Add constant for intercept to improve prediction performance
    X = sm.add_constant(X)

    # Initial model
    model = sm.OLS(y, X).fit()

    # DataFrame to store results
    results = pd.DataFrame(columns=['Eliminated Feature', 'P-Value', 'Adjusted R2', 'AIC', 'BIC'])

    # Keep track of features
    features = list(X.columns)

    # Track steps to print in OLS summary
    step = 0

    # Loop until max p-val is below threshold
    while True:
        # Get p-values and check for the maximum
        p_values = model.pvalues
        max_p_value = p_values.max()

        # If the max p-value is below the threshold break out of loop
        if max_p_value < threshold:
            break

        # Feature to be removed
        feature_to_remove = p_values.idxmax()

        # Grab data about the feature that will be removed to display later
        new_row = {
            'Eliminated Feature': feature_to_remove,
            'P-Value': round(max_p_value, 3),
            'Adjusted R2': round(model.rsquared_adj, 3),
            'AIC': round(model.aic, 3),
            'BIC': round(model.bic, 3)
        }

        # Store soon to be removed feature
        results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

        # Print the OLS summary
        print_ols_summary(model, step)

        # Drop the feature
        features.remove(feature_to_remove)
        X = X[features]

        # Fit the new model with eliminated feature
        model = sm.OLS(y, X).fit()

        # Increment the step
        step += 1

    # Filter out const col
    final_features = list(filter(lambda f: f != 'const', features))

    # Final results
    # Rounded for neatness
    final_results = {
        'Final Features': final_features,
        'Final Adjusted R2': round(model.rsquared_adj, 3),
        'Final AIC': round(model.aic, 3),
        'Final BIC': round(model.bic, 3)
    }

    # Return the eliminated features, the final results & the final model
    return results, final_results, model

# Define features as the numerical features
features = carseat_encoded.drop(['Sales', "ShelveLoc_Good", "ShelveLoc_Medium", "Urban_Yes", "US_Yes"], axis=1)
# Target is sales
target = carseat_encoded['Sales']

# Call regression function
elimination_results, final_results, reg_final_model = backward_stepwise_regression(features, target)

# Display the elimination process table
print("Question 2 Part A:\nElimination Process:")
print(tabulate(elimination_results, headers='keys', tablefmt='pretty', floatfmt=".2f"))

# Table for final features
final_table = [
        ["Final Selected Features", ', '.join(final_results['Final Features'])],
        ["Final Adjusted R2", f"{final_results['Final Adjusted R2']:.2f}"],
        ["Final AIC", f"{final_results['Final AIC']:.2f}"],
        ["Final BIC", f"{final_results['Final BIC']:.2f}"]
    ]

# Display final features table
print("\nFinal Model Results:")
print(tabulate(final_table, headers=['Metric', 'Value'], tablefmt='pretty'))

# Question 2 Part C

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=5805, shuffle=True)

# Make sure the features align between model & test dataset
X_test_final = x_test[final_results['Final Features']]

# Add a constant for the intercept in the test set
X_test_final = sm.add_constant(X_test_final)

# Make prediction
reg_predictions = reg_final_model.predict(X_test_final)

# Compare actual vs predicted values
comparison_df = pd.DataFrame({
        'Actual Sales': y_test,
        'Predicted Sales': round(reg_predictions, 4)
    })

# Display table
print("\nComparison of Actual vs Predicted Sales:")
print(tabulate(comparison_df, headers='keys', tablefmt='pretty', floatfmt=".2f"))

# Plotting original actual values vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(original_target_test.values, label='Original Sales', color='blue', linewidth=2)
plt.plot(reg_predictions.values, label='Predicted Sales', color='orange', linewidth=2)
plt.title('Original Actual Sales vs Predicted Sales')
plt.xlabel('Index')
plt.ylabel('Sales')
plt.legend()
plt.grid()
plt.show()

# Question 2 Part D

# Calculate mean squared error
regression_mse = mean_squared_error(original_target_test, reg_predictions)

# Display the MSE
print(f"Question 2 Part D:\nMean Squared Error (MSE): {regression_mse:.2f}")

# Question 3 Part A

# Only grab numerical cols
numerical_data = carseat_encoded.drop(["ShelveLoc_Good", "ShelveLoc_Medium", "Urban_Yes", "US_Yes"], axis=1)

# Standardize the data
standardized_data = standardized(numerical_data)

# Calculate the covariance matrix
cov_matrix = np.cov(standardized_data.T)

# Calculate the e-values & e-vectors
evals, evecs = np.linalg.eig(cov_matrix)

# Sort them, so they match
sorted_indices = np.argsort(evals)[::-1]
sorted_eigenvalues = evals[sorted_indices]
sorted_eigenvectors = evecs[:, sorted_indices]

# Calculate explained variance
explained_variance_ratio = sorted_eigenvalues / np.sum(sorted_eigenvalues)

# Calculate cumulative variance
cumulative_variance = np.cumsum(explained_variance_ratio)

# Check which index is needed to get above 95%
# Add 1 to get the num of needed features
num_needed_features = np.argmax(cumulative_variance >= 0.95) + 1

# Display num needed features
print(f"Question 3 Part A:\nNumber of features needed to explain 95% of variance: {num_needed_features}")

# Question 3 Part B

# Plot the num features vs cumulative variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
plt.title('Cumulative Explained Variance by Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('Cumulative Explained Variance')
plt.xticks(range(1, len(cumulative_variance) + 1))
plt.legend()
plt.grid()
plt.show()

# Question 3 Part C

# Same plot as before, but add in a vertical line showing where cumulative variance crosses over 95%
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
plt.axvline(x=num_needed_features, color='g', linestyle='--', label=f'{num_needed_features} Components')
plt.title('Cumulative Explained Variance by Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('Cumulative Explained Variance')
plt.xticks(range(1, len(cumulative_variance) + 1))
plt.legend()
plt.grid()
plt.show()

# Question 4

# Create Random Forest & fit it to the data
rf = RandomForestRegressor(n_estimators=50, random_state=5805)
rf.fit(x_train, y_train)

# Grab feature importance
feature_importance = rf.feature_importances_

# Put it into a df for plotting
feature_importance_df = pd.DataFrame({
    'Feature': features.columns,
    'Importance': feature_importance
})

# Sort it, so the graph is descending
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance from Random Forest')
plt.grid(axis='x')
plt.show()

# Question 4 Part B

# I chose this as a threshold
threshold = 0.075

# Eliminate features based on chosen threshold
# Store eliminated features & kept features
eliminated_features = feature_importance_df[feature_importance_df['Importance'] < threshold]['Feature'].tolist()
selected_features = feature_importance_df[feature_importance_df['Importance'] >= threshold]['Feature'].tolist()

# Display eliminated features
print(f"\nQuestion 4 Part B:\nEliminated Features:\n{eliminated_features}")

# Display kept/selected features
print(f"\nQuestion 4 Part B\nFinal Selected Features:\n{selected_features}")

# Question 4 Part C

# Specify features to be used for training & testing as the selected features only
x_train_selected = x_train[selected_features]
x_test_selected = x_test[selected_features]

# Add constant for improved accuracy
x_train_selected = sm.add_constant(x_train_selected)

# Create the model
rf_model = sm.OLS(y_train, x_train_selected).fit()

# Output OLS summary
print(f"\nQuestion 4 Part C:\nOLS Summary:\n{rf_model.summary()}")

# Question 4 Part D

# Add constant to test
x_test_selected = sm.add_constant(x_test_selected)

# Predict
rf_predictions = rf_model.predict(x_test_selected)

# Plot prediction vs original test values
plt.figure(figsize=(10, 6))
plt.plot(original_target_test.values, label='Original Sales', color='blue', linewidth=2)
plt.plot(rf_predictions.values, label='Predicted Sales', color='orange', linewidth=2)
plt.title('Original Actual Sales vs Predicted Sales')
plt.xlabel('Index')
plt.ylabel('Sales')
plt.grid()
plt.legend()
plt.show()

# Question 4 Part E

# Calculate Mean Squared Error
rf_mse = mean_squared_error(original_target_test, rf_predictions)

# Display the MSE
print(f"\nQuestion 4 Part E:\nMean Squared Error (MSE): {rf_mse:.2f}")

# Question 5

# Grab all relevant data for regression model for model comparison
reg_r_sq = reg_final_model.rsquared
reg_adj_r_sq = reg_final_model.rsquared_adj
reg_aic = reg_final_model.aic
reg_bic = reg_final_model.bic
# Regression MSE is stored in regression_mse

# Grab all relevant data for Random Forest model
rf_r_sq = rf_model.rsquared
rf_adj_r_sq = rf_model.rsquared_adj
rf_aic = rf_model.aic
rf_bic = rf_model.bic
# Random Forest MSE is stored in rf_mse

# Create a comparison table between the two methods/models
comparison_table = pd.DataFrame({
    'Measure': ['R-squared', 'Adjusted R-squared', 'AIC', 'BIC', 'MSE'],
    'Regression Prediction': [reg_r_sq, reg_adj_r_sq, reg_aic, reg_bic, regression_mse],
    'Random Forest Prediction': [rf_r_sq, rf_adj_r_sq, rf_aic, rf_bic, rf_mse]
})

# Display comparison table
print(f"\nQuestion 5:\nComparison of Stepwise Regression vs Random Forest:\n{comparison_table}")

# Question 6

# Grab prediction
reg_predict = reg_final_model.get_prediction(X_test_final)

# Put it into a summary_frame
reg_predict_summary = reg_predict.summary_frame(alpha=0.05)

# Define x axis indices
sample_indices = range(len(original_target_test))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(sample_indices, reg_predictions.values,
         color='blue', label='Predicted Sales', linewidth=4)
# Confidence interval
plt.fill_between(sample_indices,
                 reg_predict_summary['obs_ci_lower'],
                 reg_predict_summary['obs_ci_upper'],
                 color='lightblue',
                 alpha=0.5, label='95% Prediction Interval')
plt.title('Sales Prediction With Confidence Interval')
plt.xlabel('Number of Samples')
plt.ylabel('Sales')
plt.grid()
plt.legend()
plt.show()

# Question 7 Part A

# Define pipeline
pipeline = make_pipeline(PolynomialFeatures(), LinearRegression())

# Define param grid with range from 1-15
param_grid = {'polynomialfeatures__degree': np.arange(1, 16)}

# Perform grid search
grid_search = GridSearchCV(pipeline,
                           param_grid,
                           cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

# Define the feature as Price
grid_search_feature = carseat_encoded[['Price']]

# Define target as Sales
grid_search_target = carseat_encoded['Sales']

# Fit the model
grid_search.fit(grid_search_feature, grid_search_target)

# Grab best RMSE
# Negate it to get a pos score & round to 3 dec places
best_rmse = -round(grid_search.best_score_, 3)

# Display best RMSE
print(f"\nQuestion 7 Part A:\nBest RMSE: {best_rmse}")

# Question 7 Part B

# Grab best param/degree/n
best_param = grid_search.best_params_['polynomialfeatures__degree']

# Display it
print(f"\nQuestion 7 Part B:\nBest Polynomial Degree: {best_param}")

# Question 7 Part C

# Grab RMSE vals
rmse_values = -grid_search.cv_results_['mean_test_score']

# Grab degrees
degrees = grid_search.cv_results_['param_polynomialfeatures__degree']

# Plot RMSE vals vs degrees
plt.figure(figsize=(10, 6))
plt.plot(degrees, rmse_values, marker='o')
plt.title('RMSE vs. Polynomial Degree')
plt.xlabel('Polynomial Degree')
plt.ylabel('RMSE (Logarithmic scale)')
plt.yscale('log')
plt.xticks(degrees)
plt.grid()
plt.show()

# Question 7 Part D

# Split the dataset
grid_train, grid_test, grid_target_train, grid_target_test = train_test_split(grid_search_feature,
                                                                              grid_search_target,
                                                                              test_size=0.2,
                                                                              random_state=5805)

# Grab the best n/model
grid_best_model = grid_search.best_estimator_

# Fit the best model
grid_best_model.fit(grid_train, grid_target_train)

# Make a prediction
grid_predictions = grid_best_model.predict(grid_test)

# Plot prediction vs test vals
plt.figure(figsize=(10, 6))
plt.plot(grid_target_test.values, label='Test Sales', color='green', linewidth=2)
plt.plot(grid_predictions, label='Predicted Sales', color='orange', linewidth=2)
plt.title('Original Sales vs Predicted Sales')
plt.xlabel('Observations')
plt.ylabel('Sales')
plt.legend()
plt.grid()
plt.show()

# Question 7 Part E

# Calculate MSE
grid_mse = mean_squared_error(grid_target_test, grid_predictions)

# Display MSE
print(f"\nQuestion 7 Part E:\nMean Squared Error: {grid_mse:.3f}")