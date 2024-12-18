import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Question 2

# Define sigma space
sigma = np.linspace(0.01, 0.99, 1000)

# Cross entropy for y = 1 & y = 0
ce_y1 = -np.log(sigma)
ce_y0 = -np.log(1 - sigma)

# Plot
plt.figure(figsize=(12, 8))
plt.plot(sigma,
         ce_y1,
         label="Cross-Entropy (y=1)",
         linewidth=3)
plt.plot(sigma,
         ce_y0,
         label="Cross-Entropy (y=0)",
         linewidth=3,
         linestyle="dashed")

plt.title("Log-Loss function")
plt.xlabel("σ(x)")
plt.ylabel("Cross-Entropy Loss")
plt.legend()
plt.grid()
#plt.show()

# Question 3 Part A

x, y = make_classification(n_samples=1000,
                            n_features=2,
                            n_informative=2,
                            n_redundant=0,
                            n_clusters_per_class=2,
                            random_state=5805)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5805)

logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["Actual 0", "Actual 1"], cbar=False, linewidths=2, linecolor="black")


plt.title("Confusion Matrix Regression Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
#plt.show()

# Question 3 Part B

y_prob = logreg.predict_proba(x_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, lw=3, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("(TPR)")
plt.title("(ROC) Curve")
plt.legend(loc="lower right")
plt.grid()
#plt.show()

# Question 3 Part C

acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Question 3 Part C:\nAccuracy = {acc:.2f}\nPrecision = {precision:.2f}\nRecall = {recall:.2f}")

# Question 4

smarket = pd.read_csv("smarket.csv")
print(f"\nQuestion 4:\nSmarket Dataset Head:\n{smarket.head()}")


# Question 4 Part A

features = smarket[["Lag1", "Lag2", "Lag3", "Lag4", "Lag5", "Volume"]]
target = smarket["Direction"]

smote = SMOTE(random_state=5805)
smote_features, smote_target = smote.fit_resample(features, target)

fig, ax = plt.subplots(1, 2, figsize=(12, 8))

# Plot before SMOTE
smarket["Direction"].value_counts().plot(kind="bar", color=["blue", "red"], ax=ax[0])
ax[0].set_title("Class Distribution in the Original Dataset (Imbalanced)")
ax[0].set_xlabel("Direction")
ax[0].set_ylabel("Count")

# Plot after SMOTE
pd.Series(smote_target).value_counts().plot(kind="bar", color=["blue", "green"], ax=ax[1])
ax[1].set_title("Class Distribution After SMOTE (Balanced)")
ax[1].set_xlabel("Direction")
ax[1].set_ylabel("Count")

plt.tight_layout()
#plt.show()

# Question 4 Part B

x_train, x_test, y_train, y_test = train_test_split(smote_features, smote_target, test_size=0.2, random_state=5805, shuffle=True)

scaler = StandardScaler()
scaled_x_train = scaler.fit_transform(x_train)
scaled_x_test = scaler.transform(x_test)

train_set = pd.DataFrame(scaled_x_train, columns=features.columns)
train_set["Direction"] = y_train

test_set = pd.DataFrame(scaled_x_test, columns=features.columns)
test_set["Direction"] = y_test

print(f"\nQuestion 4 Part B:\nTrain Dataset Head:\n{train_set.head()}\nTest Dataset Head:\n{test_set.head()}")

# Question 4 Part C Subsection i

logreg = LogisticRegression(random_state=5805)

y_train_encoded = LabelEncoder().fit_transform(y_train)
y_test_encoded = LabelEncoder().fit_transform(y_test)

logreg.fit(scaled_x_train, y_train_encoded)

y_pred = logreg.predict(scaled_x_test)

train_acc = logreg.score(scaled_x_train, y_train_encoded)
test_acc = logreg.score(scaled_x_test, y_test_encoded)

print(f"\nQuestion 4 Part C Subsection i:\nTrain Dataset Accuracy = {train_acc:.2f}\nTest Dataset Accuracy = {test_acc:.2f}")


# Question 4 Part C Subsection ii

y_pred_encoded = logreg.predict(scaled_x_test)

cm = confusion_matrix(y_test_encoded, y_pred_encoded)

plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=["Predicted Down", "Predicted Up"],
            yticklabels=["Actual Down", "Actual Up"], cbar=False, linewidths=2, linecolor="black")

plt.title("Confusion Matrix Logistic Regression Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
#plt.show()

# Question 4 Part C Subsection iii

fpr, tpr, _ = roc_curve(y_test_encoded, logreg.predict_proba(scaled_x_test)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="black", linestyle="--")
plt.title("(ROC) Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid()
plt.legend()
#plt.show()

# Question 4 Part C Subsection iv

print(f"\nQuestion 4 Part C Subsection iv:\nClassification Report:\n{classification_report(y_test_encoded, y_pred_encoded)}")

# Question 4 Part D

# Question 4 Part D Subsection i
# Question 4 Part D Subsection ii
# Question 4 Part D Subsection iii
# Question 4 Part D Subsection iv
# Question 4 Part D Subsection v