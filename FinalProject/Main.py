from statistics import variance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from mlxtend.preprocessing import TransactionEncoder
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, r2_score, \
    explained_variance_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score, \
    accuracy_score, silhouette_score, adjusted_rand_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from prettytable import PrettyTable
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, label_binarize, LabelEncoder, KBinsDiscretizer
from sklearn.linear_model import LinearRegression, LogisticRegression
import statsmodels.api as sm
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.multioutput import MultiOutputClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Import dataset
dataset = pd.read_csv("data.csv")

#|**********************|
#|      Data Vis        |
#|**********************|

# Plot all personality types in bar plot
personality_types = dataset['Personality'].value_counts()

plt.figure(figsize=(12,8))
personality_types.plot(kind='bar', color='blue')
plt.title("Distribution of Personality Types")
plt.xlabel("Personality Type")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
#plt.show()

# Plot all interests in bar plot
interests = dataset['Interest'].value_counts()
plt.figure(figsize=(12,8))
interests.plot(kind='bar', color='blue')
plt.title("Distribution of Interests")
plt.xlabel("Interest")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
#plt.show()

#|**********************|
#|      Encoding        |
#|**********************|

# Mapping for personality feature
personality_mapping ={"ENFP" : 0,
                      "ESFP" : 1,
                      "INTP" : 2,
                      "INFP" : 3,
                      "ENFJ" : 4,
                      "ENTP" : 5,
                      "ESTP" : 6,
                      "ISTP" : 7,
                      "INTJ" : 8,
                      "INFJ" : 9,
                      "ISFP" : 10,
                      "ENTJ" : 11,
                      "ESFJ" : 12,
                      "ISFJ" : 13,
                      "ISTJ" : 14,
                      "ESTJ" : 15}

# Mapping for interest feature
interest_mapping = {"Unknown" : 0,
                    "Sports" : 1,
                    "Technology" : 2,
                    "Arts" : 3,
                    "Others" : 4}

# Mapping for gender feature
gender_mapping = {"Female" : 0, "Male" : 1}

# Just get the encoded personality feature
encode_dataset = pd.get_dummies(dataset, columns=["Personality"], drop_first=False)

# Store unaltered target for later use
unaltered_target = dataset["Personality"]

# Encode the dataset
dataset["Personality"] = dataset["Personality"].map(personality_mapping)
encode_dataset["Interest"] = encode_dataset["Interest"].map(interest_mapping)
dataset["Interest"] = dataset["Interest"].map(interest_mapping)

encode_dataset["Gender"] = encode_dataset["Gender"].map(gender_mapping)
dataset["Gender"] = dataset["Gender"].map(gender_mapping)

pd.set_option('display.max_columns', None)
#print(f"Dataset after Encoding Personality & Interests features:\n{encode_dataset.head()}")

#|**********************|
#|   Standardization    |
#|**********************|

# Define features
features = encode_dataset.drop(columns=[col for col in encode_dataset if 'Personality' in col])
# Define target
target = encode_dataset[[col for col in encode_dataset if 'Personality' in col]]

# Change target to int instead of true false
target = target.astype(int)

# Standardize the features
stand_features = StandardScaler().fit_transform(features)

# Change features to a dataframe
features = pd.DataFrame(stand_features, columns=features.columns)

encode_dataset["Age"] = features["Age"]
encode_dataset["Introversion Score"] = features["Introversion Score"]
encode_dataset["Sensing Score"] = features["Sensing Score"]
encode_dataset["Judging Score"] = features["Judging Score"]
encode_dataset["Thinking Score"] = features["Thinking Score"]


# Grab numerical features
num_stand_features = features[["Age",
                                     "Introversion Score",
                                     "Sensing Score",
                                     "Thinking Score",
                                     "Judging Score",]]

# Plot the pairwise scatter plots for the numerical features to see distribution
plt.figure(figsize=(12,8))
sns.pairplot(num_stand_features)
plt.suptitle('Pairwise Scatter Plots of Standardized Features', y=1.02)
plt.grid()
#plt.show()

#|**********************|
#|  Feature Selection   |
#|**********************|

# PCA Analysis

pca = PCA().fit(features)

# Get explained variance
exp_var = pca.explained_variance_ratio_

# Get cumulative explained variance
cum_exp_var = exp_var.cumsum()

# Plotting number of features for 95% variance
plt.figure(figsize=(12,8))
plt.plot(cum_exp_var,
         color='blue')
plt.axhline(y=0.95, color="red", linestyle="--", label="95% Variance")
plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid()
plt.legend()
#plt.show()

# Plot Principal Directions

# Grab components from pca
components = pca.components_

# Change to dataframe
pca_df = pca.transform(stand_features)

# Plot the principal components and their directions
plt.figure(figsize=(12,8))
plt.scatter(pca_df[:, 0], pca_df[:, 1], alpha=0.7, color='b')

for i, col in enumerate(features.columns):
    plt.arrow(0, 0, components[0, i], components[1, i],
              color='r', alpha=0.75, head_width=0.05, head_length=0.1)
    plt.text(components[0, i] * 1.2, components[1, i] * 1.2, features.columns[i], color='r', ha='center', va='center')

plt.title('PCA - Principal Directions (Eigenvectors)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
#plt.show()

# SVD Analysis

# Grab necessary information from SVD
U, sigma, Vt = np.linalg.svd(features, full_matrices=False)

# Calculate explained variance
explained_var = (sigma**2) / np.sum(sigma**2)

# Calculate cumulative variance
cumulative_var = np.cumsum(explained_var)

# Plot the variance for each component
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(sigma) + 1), explained_var, color='b', alpha=0.7)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance')
plt.title('Variance Per Principal Component')
plt.xticks(range(1, len(sigma) + 1))
#plt.show()

# Correlation Matrix

# Grab correlation matrix
cm = dataset.corr()

# Plot Correlation Matrix
plt.figure(figsize=(12,8))
sns.heatmap(cm, annot=True, cmap="coolwarm", fmt=".4f")
plt.title("Correlation Matrix")
#plt.show()

# Random Forest Feature Importance

# Split dataset
#x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=5805)

# Define Random Forest Regressor
#rf = RandomForestRegressor(n_estimators=50, random_state=5805)

# Fit the data
#rf.fit(x_train, y_train)

# Grab importances
#importances = rf.feature_importances_

# Change to dataframe
#importance_df = pd.DataFrame({
#    'Feature': features.columns,
#    'Importance': importances
#})

# Sort the features by importance
#importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Display importances
#print(f"\nRandom Forest Feature Importance:\n{importance_df}")

# VIF

# Add a constant to features
vif_dataset = sm.add_constant(features)

# Define dataframe to store VIF values
vif = pd.DataFrame()

# Add features & VIF to dataset
vif["Feature"] = vif_dataset.columns
vif["VIF"] = [variance_inflation_factor(vif_dataset.values, i) for i in range(vif_dataset.shape[1])]

# Display VIF data
print(f"\nVIF Data:\n{vif}")

# Pearson Correlation Coefficient Matrix

# Calculate Pearson Correlation Coefficient Matrix
pcorr = dataset.corr(method="pearson")

# Plot the matrix
plt.figure(figsize=(12,8))
sns.heatmap(pcorr, annot=True, cmap="coolwarm", fmt=".4f", cbar=True)

plt.title("Pearson Correlation Coefficient Matrix")
plt.tight_layout()
#plt.show()

#|**********************|
#| Regression Analysis  |
#|**********************|

lin_features = dataset.drop(columns=["Introversion Score"])

# Define target
lin_target = dataset["Introversion Score"]

lin_target = pd.DataFrame(lin_target, columns=["Introversion Score"])

# Split the dataset for Linear Regression
x_train, x_test, y_train, y_test = train_test_split(lin_features, lin_target, test_size=0.2, random_state=5805)

# # Change the trains to dataframes
# x_train = pd.DataFrame(x_train)
# y_train = pd.DataFrame(y_train)
#
# # Encode the target for train & test
# y_train_encoded = y_train.replace({True: 1, False: 0})
# y_test_encoded = y_test.replace({True: 1, False: 0})
#
# # Standardize the target
# y_train = StandardScaler().fit_transform(y_train_encoded)
# y_test = StandardScaler().fit_transform(y_test_encoded)

# OLS

ols_metrics = []

# Add a constant
lin_features_const = sm.add_constant(lin_features)

# Get OLS model for each target column
ols_model = sm.OLS(lin_target, lin_features_const).fit()

# Get P-value
p_vals = round(ols_model.pvalues, 4)

t_vals = round(ols_model.tvalues, 4)

conf_int = round(ols_model.conf_int(), 4)

# Add metrics to list
ols_metrics.append( { "R-Squared": round(ols_model.rsquared, 2),
                    "Adjusted R-Squared": round(ols_model.rsquared_adj, 2),
                    "AIC": round(ols_model.aic, 2),
                    "BIC": round(ols_model.bic, 2),
                    "F-Value": round(ols_model.fvalue, 2)})

# Define OLS table
ols_metrics_table = PrettyTable()

ols_metrics_table.field_names = ["Metric", "Value"]

# Add the rows for OLS metrics
for metric in ols_metrics[0]:
    ols_metrics_table.add_row([metric, ols_metrics[0][metric]])

# Define P-value tables
p_vals_table = PrettyTable()

p_vals_table.field_names = ["Metric", "P-Value"]

# Add the rows for P-values
for var, p_val in p_vals.items():
    p_vals_table.add_row([var, p_val])

# Define T-value table
t_vals_table = PrettyTable()

t_vals_table.field_names = ["Metric", "T-Value"]

# Add rows to T-values table
for var, t_val in t_vals.items():
    t_vals_table.add_row([var, t_val])

# Define the confidence interval table
ci_table = PrettyTable()

ci_table.field_names = ["Metric", "Lower CI", "Upper CI"]

# Loop through the confidence intervals
for i, var in enumerate(conf_int.index):
    ci_table.add_row([var, conf_int.iloc[i, 0], conf_int.iloc[i, 1]])

# Display the tables for different metrics
print(f"\nOLS Metric Summary for Introversion Score:\n{ols_metrics_table}")

print(f"\nP-values for Introversion Score:\n{p_vals_table}")

print(f"\nT-test Values For Introversion Score:\n{t_vals_table}")

print(f"\nConfidence Intervals For Introversion Score:\n{ci_table}")

def backward_stepwise_regression(x, y, threshold=0.01):

    # Add constant for intercept to improve prediction performance
    x = sm.add_constant(x)

    # Initial model
    model = sm.OLS(y, x).fit()

    # DataFrame to store results
    results = pd.DataFrame(columns=['Eliminated Feature', 'P-Value', 'Adjusted R2', 'AIC', 'BIC'])

    # Keep track of features
    features = list(x.columns)

    # Loop until max p-val is below threshold
    while True:
        # Get p-values
        p_values = model.pvalues

        # Grab max
        max_p_value = p_values.max()

        # Check if max is below threshold
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

        # Drop the feature
        features.remove(feature_to_remove)
        x = x[features]

        # Fit the new model with eliminated feature
        model = sm.OLS(y, x).fit()

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

# Store results from backwards stepwise regression
elim_results, final_stepwise_results, stepwise_model = backward_stepwise_regression(lin_features, dataset["Introversion Score"])

# Display the elimination process
print(f"\nBackwards Stepwise Regression Elimination Process:")
print(tabulate(elim_results, headers='keys', tablefmt='pretty', floatfmt=".2f"))

# Table for final features from backwards stepwise regression
final_table = [
        ["Final Selected Features", ', '.join(final_stepwise_results['Final Features'])],
        ["Final Adjusted R2", f"{final_stepwise_results['Final Adjusted R2']:.2f}"],
        ["Final AIC", f"{final_stepwise_results['Final AIC']:.2f}"],
        ["Final BIC", f"{final_stepwise_results['Final BIC']:.2f}"]
    ]

# Display final features table
print("\nFinal Backwards Stepwise Regression Model Results:")
print(tabulate(final_table, headers=['Metric', 'Value'], tablefmt='pretty'))

# Final Linear Regression Model

# Drop unnecessary features
lin_features = lin_features.drop(columns=["Gender", "Interest"])

# Split dataset again
x_train, x_test, y_train, y_test = train_test_split(lin_features, lin_target, test_size=0.2, random_state=5805)

# Define Linear Regression model
lin_reg_model = LinearRegression().fit(x_train, y_train)

intercept = lin_reg_model.intercept_[0]
coeffs = lin_reg_model.coef_[0]

eq = f"y = {intercept:.4f}"
for f, c in zip(lin_features.columns, coeffs):
    eq += f" + ({c:.4f}) * {f}"

print(f"\nFinal Linear Regression Model:\n{eq}")

# Get the prediction for train & test
lin_reg_train_pred = lin_reg_model.predict(x_train)
lin_reg_test_pred = lin_reg_model.predict(x_test)

# Metrics for Linear Regression
lin_reg_r2_score = r2_score(y_test, lin_reg_test_pred)
lin_reg_mse = mean_squared_error(y_test, lin_reg_test_pred)
lin_reg_mae = mean_absolute_error(y_test, lin_reg_test_pred)

# Display metrics
print(f"\nLinear Regression Metrics:\nR-squared: {lin_reg_r2_score: .4f}\nMSE: {lin_reg_mse : .4f}\nMAE: {lin_reg_mae : .4f}")

# Plot results of Linear Regression

plt.figure(figsize=(12,8))

plt.subplot(1,2,1)
plt.plot(y_train.values, label='Train Values', color='green', alpha=0.5)
plt.plot(lin_reg_train_pred, label='Predicted Values', color='orange', alpha=0.5)
plt.title("Linear Regression Predicted Train vs Actual Train")
plt.xlabel("Actual Train")
plt.ylabel("Predicted Train")
plt.legend()

plt.subplot(1,2,2)
plt.plot(y_test.values, label="Test Values", color="blue", alpha=0.5)
plt.plot(lin_reg_test_pred, label='Predicted Values', color="red", alpha=0.5)
plt.title("Linear Regression Predicted Test vs Actual Test")
plt.xlabel("Actual Test")
plt.ylabel("Predicted Test")
plt.legend()

plt.tight_layout()
#plt.show()


#|**********************|
#|   Classification     |
#|**********************|

# Function to display performance metrics
def compute_classification_metrics(model, x_test, y_test, average="micro"):

    # Get prediction
    y_pred = model.predict(x_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Precision
    prec = precision_score(y_test, y_pred, average=average)

    # Sensitivity/Recall
    recall = recall_score(y_test, y_pred, average=average)

    # Specificity
    specificity = []
    for i in range(cm.shape[0]):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        specificity.append(tn / (tn + fp))
    specificity = np.mean(specificity)

    # F-score
    f_score = f1_score(y_test, y_pred, average=average)

    # All metrics
    metrics = {
        "Confusion Matrix": cm,
        "Precision": round(prec, 4),
        "Recall": round(recall, 4),
        "Specificity": round(specificity, 4),
        "F-Score": round(f_score, 4),
    }

    # Return metrics
    return metrics

# Stratified K-fold cross validation
def compute_k_fold_cross_validation(model, features, target, n_splits=5):

    # Define the Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)

    # Lists to store results
    precisions = []
    recalls = []
    specificities = []
    f_scores = []

    # Loop through
    for train_index, test_index in skf.split(features, target):
        # Get features
        x_train, x_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        # Train the model
        model.fit(x_train, y_train)

        # Use the compute_classification_metrics function to get the metrics for this fold
        metrics = compute_classification_metrics(model, x_test, y_test)

        # Append metrics to the lists
        precisions.append(metrics['Precision'])
        recalls.append(metrics['Recall'])
        specificities.append(metrics['Specificity'])
        f_scores.append(metrics['F-Score'])

    # Average metrics across all folds
    avg_metrics = {
        'Average Precision': round(np.mean(precisions), 4),
        'Average Recall': round(np.mean(recalls), 4),
        'Average Specificity': round(np.mean(specificities), 4),
        'Average F-Score': round(np.mean(f_scores), 4)
    }

    return avg_metrics

# Function to plot the Confusion Matrix for each model
def plot_cm(cm, name, y_test):

    # Plot using heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm,
                annot=True,
                cmap="coolwarm",
                fmt=",.0f",
                xticklabels=np.unique(y_test),
                yticklabels=np.unique(y_test))

    plt.title(f"Confusion Matrix for {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

# Function to plot ROC Curves for multiple models
def plot_roc_curves(models, x_test, y_test, names):
    plt.figure(figsize=(10,8))

    y_test_binarize = label_binarize(y_test, classes=np.unique(y_test))

    # Loop through models
    for model, name in zip(models, names):

        #Get probability
        y_prob = model.predict_proba(x_test)

        fpr, tpr, _ = roc_curve(y_test_binarize.ravel(), y_prob.ravel())

        auc = roc_auc_score(y_test_binarize, y_prob, average="micro")

        # Plot
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")

    # Plot diagonal line
    plt.plot([0, 1], [0, 1],
             color="red",
             linestyle="--",
             label="Random (AUC = 0.50)")
    plt.title("ROC Curves For Classification Models")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

# Decision Tree

# Split the dataset using the unaltered target feature
x_train, x_test, y_train, y_test = train_test_split(features, unaltered_target, test_size=0.2, random_state=5805)

# Define Decision Tree
#dt = DecisionTreeClassifier(random_state=5805, splitter="best")
# dt = DecisionTreeClassifier(random_state=5805,
#                            splitter="best",
#                            ccp_alpha=0,
#                            criterion="entropy",
#                            max_depth=20,
#                            max_features="log2",
#                            min_samples_split=3)
#
# # Define param_grid
# param_grid = {
#     'criterion': ['gini', 'entropy'],
#     'max_depth': [20, 25],
#     'min_samples_split': [2, 3],
#     'max_features': ['sqrt', 'log2'],
#     'ccp_alpha': [0, 0.01]
# }
#
#
# # Grid search for best parameters
# #grid_search = GridSearchCV(estimator=dt,
# #                           param_grid=param_grid,
# #                           cv=5,
# #                           n_jobs=-1,
# #                           scoring='accuracy',)
#
# print("\nSearching for best parameters for Decision Tree...\n")
# #grid_search.fit(x_train, y_train)
#
# #print(f"\nBest Parameters for Decision Tree:\n {grid_search.best_params_}")
#
# # Get best model from grid search
# #dt_best_model = grid_search.best_estimator_
# dt_best_model = dt.fit(x_train, y_train)
#
# #Calculate metrics
# dt_metrics = compute_classification_metrics(dt_best_model, x_test, y_test)
#
# #Calculate stratified k-fold metrics
# dt_k_fold_metrics = compute_k_fold_cross_validation(dt_best_model, x_test, y_test)

# # Logistic Regression

# Define Logistic Regression

# y_train_enc = LabelEncoder().fit_transform(y_train)
#
# #log_reg = LogisticRegression()
# log_reg = LogisticRegression(C=1, max_iter=100, solver="lbfgs", penalty="l2")
#
# # Define param grid
# param_grid = {
#     "C": [0.01, 0.1, 1],
#     "penalty": ["l2"],
#     "solver": ["lbfgs"],
#     "max_iter": [100, 150]
# }
#
# # Perform grid search
# #grid_search = GridSearchCV(estimator=lr,
# #                           param_grid=param_grid,
# #                           cv=5,
# #                           n_jobs=-1,
# #                           scoring="accuracy",)
#
# print(f"\nSearching for best parameters for Logistic Regression...\n")
#
# #grid_search.fit(x_train, y_train)
#
# #print(f"Best Parameters for Logistic Regression:\n{grid_search.best_params_}")
# log_reg.fit(x_train, y_train)
# #Get best model from grid search
# #lr_best_model = grid_search.best_estimator_
# log_reg_best_model = log_reg
#
# #Grab metrics
# log_reg_metrics = compute_classification_metrics(log_reg_best_model, x_test, y_test)
#
# #Grab stratified k-fold metrics
# log_reg_k_fold_metrics = compute_k_fold_cross_validation(log_reg_best_model, x_test, y_test)

# KNN
#
# Range to search for K values from
# k_vals = range(1,25)
#
# # Store accuracy scores from different k-values
# acc_scores = []
#
# # Loop through k-values
# for k in k_vals:
#     # Define KNN for each k-value
#     knn = KNeighborsClassifier(n_neighbors=k)
#
#     # Fit the model
#     knn.fit(x_train, y_train)
#
#     # Get the prediction
#     y_pred = knn.predict(x_test)
#
#     # Get accuracy
#     acc_scores.append(accuracy_score(y_test, y_pred))
#
# #Elbow plot for accuracy for each k-value
# plt.figure(figsize=(10, 8))
# plt.plot(k_vals,
#          acc_scores,
#          linestyle='-',
#          color="blue",
#          label='Accuracy')
# plt.title("Elbow Plot For Optimal K")
# plt.xlabel("K")
# plt.ylabel("Accuracy")
# plt.xticks(k_vals)
# plt.grid()
# #plt.show()
#
# #Grab optimal k-value from list
# opt_k_val = k_vals[np.argmax(acc_scores)]
#
# print(f"\nOptimal K is: {opt_k_val}\n")

#Get best model based on optimal k
# knn_best_model = KNeighborsClassifier(n_neighbors=19)
#
# #Fit best model
# knn_best_model.fit(x_train, y_train)
#
# #Get metrics
# knn_metrics = compute_classification_metrics(knn_best_model, x_test, y_test)
#
# #Get stratified k-fold metrics
# knn_k_fold_metrics = compute_k_fold_cross_validation(knn_best_model, x_test, y_test)

# # SVM
#
# Function to grid search for different kernels
# def svm_grid_search(x_train, y_train, param_grid):
#     print(f"\nSearching for best parameters for SVM...\n")
#
#     # Grid search
#     grid_search = GridSearchCV(SVC(),
#                                param_grid,
#                                cv=5,
#                                n_jobs=-1,
#                                scoring="accuracy",)
#
#     # Fit the model
#     grid_search.fit(x_train, y_train)
#
#     # Return grid search
#     return grid_search
#
# # Define base param grid each kernel should include
# param_grid = {
#     "C": [0.1, 1, 1.5],
#     "probability": [True]
# }
#
# # Copy base param grid and add to it
# param_grid_lin = param_grid.copy()
# param_grid_lin["kernel"] = ["linear"]
#
# # Copy base param grid and add to it for this kernel
# param_grid_poly = param_grid.copy()
# param_grid_poly["kernel"] = ["poly"]
# param_grid_poly["degree"] = [2, 3]
# param_grid_poly["gamma"] = ["auto"]
#
# # Copy base param grid and add to it for this kernel
# param_grid_rb = param_grid.copy()
# param_grid_rb["kernel"] = ["rbf"]
# param_grid_rb["gamma"] = ["scale", "auto"]
#
# # Grid search for linear kernel
# #grid_search_lin = svm_grid_search(x_train, y_train, param_grid_lin)
# #print(f"\nSVM Best Parameters for Linear Kernel:\n{grid_search_lin.best_params_}")
# # Get best model
# #lin_best_model = grid_search_lin.best_estimator_
# lin_best_model = SVC(C=0.1, kernel="linear", probability=True)
# lin_best_model.fit(x_train, y_train)
# # Get metrics
# lin_metrics = compute_classification_metrics(lin_best_model, x_test, y_test)
#
# # Grid search for polynomial kernel
# #grid_search_poly = svm_grid_search(x_train, y_train, param_grid_poly)
# #print(f"\nSVM Best Parameters for Polynomial Kernel:\n{grid_search_poly.best_params_}")
# # Get best model
# poly_best_model = SVC(C=1, degree=3, gamma="auto", kernel="poly", probability=True)
# poly_best_model.fit(x_train, y_train)
# # Get metrics
# poly_metrics = compute_classification_metrics(poly_best_model, x_test, y_test)
#
# # Grid search for RBF kernel
# #grid_search_rbf = svm_grid_search(x_train, y_train, param_grid_rbf)
# #print(f"\nSVM Best Parameters for RBF Kernel:\n{grid_search_rbf.best_params_}")
# # Get best model
# rbf_best_model = SVC(C=1, gamma="auto", kernel="rbf", probability=True)
# rbf_best_model.fit(x_train, y_train)
# # # Get metrics
# rbf_metrics = compute_classification_metrics(rbf_best_model, x_test, y_test)
# #
# # # Define table headers for SVM kernel comparison
# # svm_table_headers = ["Metric", "Linear Kernel", "Polynomial Kernel", "RBF Kernel"]
# #
# # # Define table
# # svm_table = PrettyTable(svm_table_headers)
# #
# # # Add rows for comparison metrics
# # svm_table.add_row(["Precision", lin_metrics["Precision"], poly_metrics["Precision"], rbf_metrics["Precision"]])
# # svm_table.add_row(["Recall", lin_metrics["Recall"], poly_metrics["Recall"], rbf_metrics["Recall"]])
# # svm_table.add_row(["Specificity", lin_metrics["Specificity"], poly_metrics["Specificity"], rbf_metrics["Specificity"]])
# # svm_table.add_row(["F-Score", lin_metrics["F-Score"], poly_metrics["F-Score"], rbf_metrics["F-Score"]])
# #
# # # Display comparison table
# # print(f"\nSVM Different Kernel Metrics Table\n{svm_table}")
# #
# # # Best kernel is RBF from grid search & performance metrics
# svm_best_model = rbf_best_model
#
# # Get metrics
# svm_metrics = rbf_metrics
#
# # Get stratified k-fold metrics
# svm_k_fold_metrics = compute_k_fold_cross_validation(svm_best_model, x_test, y_test)

# # Naive Bayes
#
# # Define the model
# nb = GaussianNB()
#
# # Define param grid
# param_grid = {
#     "var_smoothing": [1e-3, 1e-1, 1e0, 1e1, 1e2]
# }
#
# print(f"\nSearching for best parameters for Naive Bayes...\n")
#
# # Grid search for best parameters
# grid_search = GridSearchCV(estimator=nb,
#                            param_grid=param_grid,
#                            cv=5,
#                            n_jobs=-1,
#                            scoring="accuracy",)
#
# # Fit the model
# grid_search.fit(x_train, y_train)
#
# print(f"Best parameters for Naive Bayes:\n{grid_search.best_params_}")
#
# # Get best model
# nb_best_model = grid_search.best_estimator_
#
# # Get metrics
# nb_metrics = compute_classification_metrics(nb_best_model, x_test, y_test)
#
# # Get stratified k-fold metrics
# nb_k_fold_metrics = compute_k_fold_cross_validation(nb_best_model, x_test, y_test)

# Random Forest

# Define the model
#rf_model = RandomForestClassifier(random_state=5805, oob_score=False)
# rf_model = RandomForestClassifier(random_state=5805,
#                                  oob_score=False,
#                                  n_estimators=100,
#                                  criterion="entropy",
#                                  max_features="sqrt",
#                                  max_depth=20,
#                                  min_samples_split=5,
#                                  min_samples_leaf=2)
# #Define param grid
# param_grid = {
#     'n_estimators': [100, 150],
#     'criterion': ['gini', 'entropy'],
#     'max_depth': [20, 30],
#     'min_samples_split': [5, 7],
#     'min_samples_leaf': [2, 4],
#     'max_features': ['sqrt', 'log2']
# }
#
# #Grid search
# #grid_search = GridSearchCV(estimator=rf_model,
# #                         param_grid=param_grid,
# #                          cv=5,
# #                          n_jobs=-1,
# #                          scoring='accuracy',)
#
# print(f"\nSearching for best parameters for Random Forest...\n")
#
# #Fit the model
# #grid_search.fit(x_train, y_train)
#
# #print(f"\nBest Parameters for Random Forest:\n {grid_search.best_params_}")
#
# #Get best model
# #rf_best_model = grid_search.best_estimator_
# rf_model.fit(x_train, y_train)
# rf_best_model = rf_model
# #Get metrics
# rf_metrics = compute_classification_metrics(rf_best_model, x_test, y_test)
#
# #Get stratified k-fold metrics
# rf_k_fold_metrics = compute_k_fold_cross_validation(rf_model, x_test, y_test)

# Neural Network

# Define the model
# nn = MLPClassifier(random_state=5805, early_stopping=True)
#
# #Define param grid
# param_grid = {
#     "hidden_layer_sizes": [(100,100), (50,50,50), (100,100,100)],
#     #"activation": ["logistic", "tanh", "relu"],
#     "activation": ["relu"],
#     "solver": ["adam"],
#     "alpha": [0.01, 0.1],
#     "learning_rate": ["adaptive"],
#     "max_iter": [4000, 5000],
# }
#
# print(f"\nSearching for best parameters for Neural Network...\n")
#
# #Grid search
# #grid_search = GridSearchCV(estimator=nn,
# #                           param_grid=param_grid,
# #                           cv=5,
# #                           n_jobs=-1,
# #                           scoring="accuracy")
#
# #Fit the model
# #grid_search.fit(x_train, y_train)
#
# #print(f"Best Parameters for Neural Network:\n{grid_search.best_params_}")
#
# #Get best model
# #nn_best_model = grid_search.best_estimator_
# nn_best_model = MLPClassifier(random_state=5805,
#                               early_stopping=True,
#                               activation="relu",
#                               alpha=0.1,
#                               hidden_layer_sizes=(100,100,100),
#                               learning_rate="adaptive",
#                               max_iter=4000,
#                               solver="adam")
#
# nn_best_model.fit(x_train, y_train)
# #Get metrics
# nn_metrics = compute_classification_metrics(nn_best_model, x_test, y_test)
#
# #Get stratified k-fold metrics
# nn_k_fold_metrics = compute_k_fold_cross_validation(nn_best_model, x_test, y_test)

# Display table with metrics from all different classifiers

# Define headers for classifier metrics table
# classifier_metrics_table_headers = ["Metric",
#                                     "Decision Tree",
#                                     "Logistic Regression",
#                                     "KNN",
#                                     "SVM",
#                                     "Naive Bayes",
#                                     "Random Forest",
#                                     "Neural Network"]

# # Define classifier metrics table
# classifier_metrics_table = PrettyTable(classifier_metrics_table_headers)
#
# # Add rows
# classifier_metrics_table.add_row(["Precision", dt_metrics["Precision"], log_reg_metrics["Precision"], knn_metrics["Precision"], svm_metrics["Precision"], nb_metrics["Precision"], rf_metrics["Precision"], nn_metrics["Precision"]])
# classifier_metrics_table.add_row(["Recall", dt_metrics["Recall"], log_reg_metrics["Recall"], knn_metrics["Recall"], svm_metrics["Recall"], nb_metrics["Recall"], rf_metrics["Recall"], nn_metrics["Recall"]])
# classifier_metrics_table.add_row(["Specificity", dt_metrics["Specificity"], log_reg_metrics["Specificity"], knn_metrics["Specificity"], svm_metrics["Specificity"], nb_metrics["Specificity"], rf_metrics["Specificity"], nn_metrics["Specificity"]])
# classifier_metrics_table.add_row(["F-Score", dt_metrics["F-Score"], log_reg_metrics["F-Score"], knn_metrics["F-Score"], svm_metrics["F-Score"], nb_metrics["F-Score"], rf_metrics["F-Score"], nn_metrics["F-Score"]])
#
# # Display table
# print(f"\nClassifier Metrics Table\n{classifier_metrics_table}")
#
# # Repeat process for stratified k-fold table
# k_fold_table = PrettyTable(classifier_metrics_table_headers)
#
# # Add rows
# k_fold_table.add_row(["Avg Precision", dt_k_fold_metrics["Average Precision"], log_reg_k_fold_metrics["Average Precision"], knn_k_fold_metrics["Average Precision"], svm_k_fold_metrics["Average Precision"], nb_k_fold_metrics["Average Precision"], rf_k_fold_metrics["Average Precision"], nn_k_fold_metrics["Average Precision"]])
# k_fold_table.add_row(["Avg Recall", dt_k_fold_metrics["Average Recall"], log_reg_k_fold_metrics["Average Recall"], knn_k_fold_metrics["Average Recall"], svm_k_fold_metrics["Average Recall"], nb_k_fold_metrics["Average Recall"], rf_k_fold_metrics["Average Recall"], nn_k_fold_metrics["Average Recall"]])
# k_fold_table.add_row(["Avg Specificity", dt_k_fold_metrics["Average Specificity"], log_reg_k_fold_metrics["Average Specificity"], knn_k_fold_metrics["Average Specificity"], svm_k_fold_metrics["Average Specificity"], nb_k_fold_metrics["Average Specificity"], rf_k_fold_metrics["Average Specificity"], nn_k_fold_metrics["Average Specificity"]])
# k_fold_table.add_row(["Avg F-Score", dt_k_fold_metrics["Average F-Score"], log_reg_k_fold_metrics["Average F-Score"], knn_k_fold_metrics["Average F-Score"], svm_k_fold_metrics["Average F-Score"], nb_k_fold_metrics["Average F-Score"], rf_k_fold_metrics["Average F-Score"], nn_k_fold_metrics["Average F-Score"]])
#
# # Display table
# print(f"\nStratified K-Fold Classifier Metrics Table:\n{k_fold_table}")
#
# # Plot all ROC Curves in one plot
#
# #Define models
# models = [dt_best_model,
#           log_reg_best_model,
#           knn_best_model,
#           svm_best_model,
#           nb_best_model,
#           rf_best_model,
#           nn_best_model]
#
# #Define model names
# model_names = ["Decision Tree",
#                "Logistic Regression",
#                "KNN",
#                "SVM",
#                "Naive Bayes",
#                "Random Forest",
#                "Neural Network"]
#
# #Plot ROC curves
# plot_roc_curves(models, x_test, y_test, model_names)
#
# # Plot Confusion Matrices
# plot_cm(dt_metrics["Confusion Matrix"], "Decision Tree", y_test)
# plot_cm(log_reg_metrics["Confusion Matrix"], "Logistic Regression", y_test)
# plot_cm(knn_metrics["Confusion Matrix"], "KNN", y_test)
# plot_cm(svm_metrics["Confusion Matrix"], "SVM", y_test)
# plot_cm(nb_metrics["Confusion Matrix"], "Naive Bayes", y_test)
# plot_cm(rf_metrics["Confusion Matrix"], "Random Forest", y_test)
# plot_cm(nn_metrics["Confusion Matrix"], "Neural Network", y_test)

#|**********************|
#|      CLUSTERING      |
#|**********************|

# K-mean/K-mean++

k_mean_features = encode_dataset.iloc[:, :8]

# Silhouette analysis
# sil_scores = []
# inertia_vals = []
# k_range = range(2,6)
#
# print(f"\nSearching for optimal K...\n")
# for k in k_range:
#     kmeans = KMeans(n_clusters=k, init="k-means++", random_state=5805)
#
#     kmeans.fit(k_mean_features)
#
#     clusters = kmeans.labels_
#
#     sil_avg = silhouette_score(k_mean_features, clusters)
#
#     sil_scores.append(sil_avg)
#
#     inertia_vals.append(kmeans.inertia_)
#
# plt.figure(figsize=(10, 8))
#
# plt.subplot(1, 2, 1)
# plt.plot(k_range, sil_scores, color="black")
# plt.title("Silhouette Scores vs Number of Clusters")
# plt.xlabel("Number of Clusters")
# plt.ylabel("Silhouette Score")
# plt.grid()
#
#
# plt.subplot(1, 2, 2)
# plt.plot(k_range, inertia_vals, color="green")
# plt.title("Inertia vs Number of Clusters")
# plt.xlabel("Number of Clusters")
# plt.ylabel("Inertia")
# plt.grid()
# plt.tight_layout()
# plt.show()
#
# optimal_sil = k_range[np.argmax(sil_scores)]
# optimal_inertia = k_range[np.argmin(inertia_vals)]
#
# print(f"\nK-mean++ Optimal Silhouette Score (K): {optimal_sil}")
# print(f"\nK-mean++ Optimal Inertia: {optimal_inertia}")

optimal_sil = 2

kmeans_best_model = KMeans(n_clusters=optimal_sil, init="k-means++", random_state=5805)
kmeans_best_model.fit(k_mean_features)

clusters = kmeans_best_model.labels_

centers = kmeans_best_model.cluster_centers_

reduce_features = TSNE(n_components=2, random_state=5805).fit_transform(k_mean_features)

#reduce_dataset = PCA(n_components=2).fit_transform(k_mean_features)

plt.figure(figsize=(10, 8))
plt.scatter(reduce_features[:, 0], reduce_features[:, 1], c=clusters,
            cmap="viridis", s=50, alpha=0.6)

plt.title(f"K-Means++ Clustering with k={optimal_sil}")
plt.xlabel("T-SNE 1")
plt.ylabel("T-SNE 2")
plt.colorbar(label="Cluster")
plt.legend()
plt.grid()
plt.show()

# DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=3)

dbscan.fit(features)

clusters = dbscan.labels_

len_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

print(f"\nDBSCAN Number of clusters: {len_clusters}\n")
print(f"Number of outliers: {list(clusters).count(-1)}")

plt.figure(figsize=(12,8))

sns.scatterplot(x=features[:,0], y=features[:,1],hue=clusters, s=100,)

plt.title("DBSCAN Clustering Results")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.show()

#|**********************|
#|   Rule Association   |
#|**********************|

# Apriori

score_bins = KBinsDiscretizer(n_bins=5, encode="onehot", strategy="uniform")

encode_dataset["Age"] = score_bins.fit_tranform(encode_dataset[["Age"]]).toarray().argmax(axis=1)
encode_dataset["Thinking Score"] = score_bins.fit_tranform(encode_dataset[["Thinking Score"]]).toarray().argmax(axis=1)
encode_dataset["Judging Score"] = score_bins.fit_tranform(encode_dataset[["Judging Score"]]).toarray().argmax(axis=1)
encode_dataset["Sensing Score"] = score_bins.fit_tranform(encode_dataset[["Sensing Score"]]).toarray().argmax(axis=1)
encode_dataset["Introversion Score"] = score_bins.fit_tranform(encode_dataset[["Introversion Score"]]).toarray().argmax(axis=1)

apriori = apriori(encode_dataset, min_support=.6, use_colnames=True)

rules = association_rules(apriori, metric="confidence", min_threshold=.75)

print(f"\nApriori Frequent Items:\n{apriori}")

print(f"\nApriori Association Rules:\n{rules}")