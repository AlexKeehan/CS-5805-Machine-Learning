import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score, roc_curve
from sklearn import tree

# Question 1

# Define range
p = np.linspace(0, 1, 500)

# Use clip to stop values of 0 slipping through
p = np.clip(p, 1e-10, 1 - 1e-10)

# Calculate Entropy
entropy_vals = -p * np.log2(p) - (1 - p) * np.log2(1 - p)

# Calculate Gini
gini_vals = 2 * p * (1 - p)

# Plot
plt.figure(figsize=(12, 8))
plt.plot(p, entropy_vals, label="Entropy", color="green", linewidth=3)
plt.plot(p, gini_vals, label="Gini", color="red", linewidth=3)

plt.title("Entropy vs Gini Index")
plt.xlabel("P")
plt.ylabel("Magnitude")
plt.grid()
plt.legend()
plt.show()

# Question 2 & 3 were done on paper

# Question 4

# Load dataset
dataset = sns.load_dataset("titanic")

# Drop null values
dataset.dropna(how="any", inplace=True)

# Specify numerical features
numerical_features = ["age", "fare", "pclass", "sibsp", "parch"]

# Define target & dependent features
features = dataset[numerical_features]
target = dataset["survived"]

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=5805, stratify=target)

# Define Decision Tree Model & fit it to data
dt_model = DecisionTreeClassifier(random_state=5805).fit(x_train, y_train)

# Get predictions for train & test
dt_train_pred = dt_model.predict(x_train)
dt_test_pred = dt_model.predict(x_test)

# Get Accuracy for train & test
dt_train_acc = accuracy_score(y_train, dt_train_pred)
dt_test_acc = accuracy_score(y_test, dt_test_pred)

# Print accuracies for training & test for un-pruned DT model
print(f"Question 4 Train & Test Accuracies:\nTrain Acc: {dt_train_acc : .2f}\nTest Acc: {dt_test_acc : .2f}")

# Use pretty print to display the parameters for this DT model
pprint.pprint(f"Question 4 Decision Tree Parameters:{dt_model.get_params()}")

# Plot the tree
plt.figure(figsize=(15,10))
tree.plot_tree(dt_model,
               filled=True,
               feature_names=numerical_features,
               class_names=["Not Survived", "Survived"],
               rounded=True)
plt.title("No Pruning Decision Tree")
plt.show()

#Question 5

# Define parameters to search on
tuned_parameters = [{
    "max_depth": [3, 5, 7, 10, 15, 20],
    "min_samples_split": [2, 10, 15, 20, 30, 35, 40],
    "min_samples_leaf": [1, 3, 5, 7, 10, 20],
    "criterion": ["gini", "entropy", "log_loss"],
    "splitter": ["best", "random"],
    "max_features": ["sqrt", "log2", None]
}]

# Define pre-prune DT model
pre_prune_dt_model = DecisionTreeClassifier(random_state=5805)

# Use grid search to find optimal params
grid_search = GridSearchCV(estimator=pre_prune_dt_model,
                           param_grid=tuned_parameters,
                           cv=5,
                           scoring="accuracy",
                           n_jobs=-1)

# Fit the data
grid_search.fit(x_train, y_train)

# Use pretty print to display parameters for this pre-pruned DT model
print("\n")
pprint.pprint(f"Question 5 Best Parameters Found By Grid Search: {grid_search.best_params_}")

# Grab best pre-prune model from grid search
best_pre_prune_dt_model = grid_search.best_estimator_

# Get predictions for train & test
pre_prune_dt_model_train_pred = best_pre_prune_dt_model.predict(x_train)
pre_prune_dt_model_test_pred = best_pre_prune_dt_model.predict(x_test)

# Get accuracies for train & test
pre_prune_dt_model_train_acc = accuracy_score(y_train, pre_prune_dt_model_train_pred)
pre_prune_dt_model_test_acc = accuracy_score(y_test, pre_prune_dt_model_test_pred)

# Display accuracies for train & test for this pre-prune DT model
print(f"\nQuestion 5 Train And Test Accuracies:\n"
      f"Train Acc: {pre_prune_dt_model_train_acc : .2f}\n"
      f"Test Acc: {pre_prune_dt_model_test_acc : .2f}")

# Plot the pre-pruned tree
plt.figure(figsize=(15,10))
tree.plot_tree(best_pre_prune_dt_model,
               filled=True,
               feature_names=numerical_features,
               class_names=["Not Survived", "Survived"],
               rounded=True)
plt.title("Pre-Pruned Decision Tree")
plt.show()

#Question 6

# Define post-prune DT model
post_prune_dt_model = DecisionTreeClassifier(random_state=5805).fit(x_train, y_train)

# Grab best path
best_path = post_prune_dt_model.cost_complexity_pruning_path(x_train, y_train)

# Grab alphas from best path
alphas = best_path.ccp_alphas

# Define lists to store train & test accuracies
train_accs = []
test_accs = []

# Loop through alphas and grab accuracies
for alpha in alphas:
    # Prune the tree with the given alpha
    dt_pruned = DecisionTreeClassifier(random_state=5805, ccp_alpha=alpha).fit(x_train, y_train)

    # Grab predictions for train & test for different alphas
    train_pred = dt_pruned.predict(x_train)
    test_pred = dt_pruned.predict(x_test)

    # Add accuracy to lists for each alpha
    train_accs.append(accuracy_score(y_train, train_pred))
    test_accs.append(accuracy_score(y_test, test_pred))

# Best alpha is the one that offers the highest accuracy
best_alpha = alphas[np.argmax(test_accs)]

# Print all possible alphas & the best one
print(f"\nQuestion 6 All Possible Alphas:\n{alphas}\n\nOptimal Alpha: {best_alpha : .4f}")

# Define the post-prune DT model with the best alpha
post_prune_dt_final_model = DecisionTreeClassifier(random_state=5805, ccp_alpha=best_alpha).fit(x_train, y_train)

# Get predictions for train & test
post_prune_dt_train = post_prune_dt_final_model.predict(x_train)
post_prune_dt_test = post_prune_dt_final_model.predict(x_test)

# Get accuracies for train & test
train_acc_final = accuracy_score(y_train, post_prune_dt_train)
test_acc_final = accuracy_score(y_test, post_prune_dt_test)

# Display accuracies for train & test
print(f"\nQuestion 6 Optimal Alpha Accuracy:\n"
      f"Train Acc: {train_acc_final: .2f}\n"
      f"Test Acc: {test_acc_final: .2f}")

# Plot the tree
plt.figure(figsize=(15,10))
tree.plot_tree(post_prune_dt_final_model,
               filled=True,
               feature_names=numerical_features,
               class_names=["Not Survived", "Survived"],
               rounded=True)

plt.title(f"Final Pruned Decision Tree")
plt.show()

# Question 7

# Define Logistic Regression model
lr_model = LogisticRegression(random_state=5805).fit(x_train, y_train)

# Get predictions for train & test
lr_train_pred = lr_model.predict(x_train)
lr_test_pred = lr_model.predict(x_test)

# Get accuracies for train & test
lr_train_acc = accuracy_score(y_train, lr_train_pred)
lr_test_acc = accuracy_score(y_test, lr_test_pred)

# Display accuracies for train & test
print(f"\nQuestion 7 Logistic Regression Model Accuracies:\nTrain Acc: {lr_train_acc : .2f}\nTest Acc: {lr_test_acc : .2f}")

# Question 8

# Function to compute metrics for the different models
def compute_perf_metrics(model, x_train, y_train, x_test, y_test):

    # Get predictions for train & test for each model
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    # Get accuracy for train & test for each model
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    # Get recall for train & test for each model
    recall_train = recall_score(y_train, train_pred)
    recall_test = recall_score(y_test, test_pred)

    # Get Confusion Matrix for train & test for each model
    cm_train = confusion_matrix(y_train, train_pred)
    cm_test = confusion_matrix(y_test, test_pred)

    # Get AUC for train & test for each model
    auc_train = roc_auc_score(y_train, model.predict_proba(x_train)[:, 1])
    auc_test = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])

    # Get false positive rate & true positive rate for train & test for each model
    # Used for ROC curve
    # Threshold is not needed
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(x_test)[:, 1])

    # Return dict of all performance metrics for the given model
    return {
        "acc_train": train_acc,
        "acc_test": test_acc,
        "cm_train": cm_train,
        "cm_test": cm_test,
        "recall_train": recall_train,
        "recall_test": recall_test,
        "auc_train": auc_train,
        "auc_test": auc_test,
        "fpr": fpr,
        "tpr": tpr
    }

# Call performance metrics function for each model
pre_prune_metrics = compute_perf_metrics(best_pre_prune_dt_model, x_train, y_train, x_test, y_test)
post_prune_metrics = compute_perf_metrics(post_prune_dt_final_model, x_train, y_train, x_test, y_test)
lr_metrics = compute_perf_metrics(lr_model, x_train, y_train, x_test, y_test)

# Put the metrics into a Dataframe for display purposes
perf_df = pd.DataFrame({
    "Model": ["DT Pre-pruned", "DT Post-pruned", "Logistic Regression"],
    "Train Acc": [pre_prune_metrics["acc_train"], post_prune_metrics["acc_train"], lr_metrics["acc_train"]],
    "Test Acc": [pre_prune_metrics["acc_test"], post_prune_metrics["acc_test"], lr_metrics["acc_test"]],
    "Train Recall": [pre_prune_metrics["recall_train"], post_prune_metrics["recall_train"], lr_metrics["recall_train"]],
    "Test Recall": [pre_prune_metrics["recall_test"], post_prune_metrics["recall_test"], lr_metrics["recall_test"]],
    "Train AUC": [pre_prune_metrics["auc_train"], post_prune_metrics["auc_train"], lr_metrics["auc_train"]],
    "Test AUC": [pre_prune_metrics["auc_test"], post_prune_metrics["auc_test"], lr_metrics["auc_test"]]
})

# Make sure all rows of the DF are displayed
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Display the performance metrics DF
print(f"\nQuestion 8 Performance Metrics for Different Models:\n{perf_df}")

# Function to plot the Confusion Matrix for each model
def plot_cm(cm, model):
    # Plot using heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm,
                annot=True,
                cmap="coolwarm",
                xticklabels=["Pred: Neg", "Pred: Pos"],
                yticklabels=["True: Neg", "True: Pos"])

    plt.title(f"Confusion Matrix for {model}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

# Plot confusion matrices for each model
plot_cm(pre_prune_metrics["cm_test"], "DT Pre-pruned")
plot_cm(post_prune_metrics["cm_test"], "DT Post-pruned")
plot_cm(lr_metrics["cm_test"], "Logistic Regression")


# Plot ROC curves for each model in same plot
# Have label = AUC for each respective model
plt.figure(figsize=(12,8))

# ROC curve for Pre-Prune Model
plt.plot(pre_prune_metrics["fpr"],
         pre_prune_metrics["tpr"],
         label=f"DT Pre-pruned (AUC = {pre_prune_metrics['auc_test']:.2f})",
         color="green")

# ROC curve for Post-Prune Model
plt.plot(post_prune_metrics["fpr"],
         post_prune_metrics["tpr"],
         label=f"DT Post-pruned (AUC = {post_prune_metrics['auc_test']:.2f})",
         color="blue")

# ROC curve for Logistic Regression Model
plt.plot(lr_metrics["fpr"],
         lr_metrics["tpr"],
         label=f"Logistic Regression (AUC = {lr_metrics['auc_test']:.2f})",
         color="orange")

# Plot diagonal line
plt.plot([0, 1], [0, 1],
         linestyle="--",
         color="red",
         label="Baseline")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Comparisons")
# Lower right is best place for legend in this kind of graph
# Because hopefully the ROC curve won't be down there
plt.legend(loc="lower right")
plt.grid()
plt.show()
