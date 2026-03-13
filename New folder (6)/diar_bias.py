# =====================================
# 1. Import Libraries
# =====================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import DisparateImpactRemover


# =====================================
# 2. Load Dataset
# =====================================

df = pd.read_csv("adult.csv")

print("Dataset shape:", df.shape)
print(df.head())


# =====================================
# 3. Data Preprocessing
# =====================================

# Replace ? with missing values
df.replace("?", np.nan, inplace=True)

# Drop missing rows
df.dropna(inplace=True)

# One-hot encoding for categorical columns
df = pd.get_dummies(df, drop_first=True)

print("\nAfter Encoding:")
print(df.head())


# =====================================
# 4. Define Target Variable
# =====================================

# After encoding income becomes income_>50K
target_column = "income_>50K"

X = df.drop(target_column, axis=1)
y = df[target_column]


# =====================================
# 5. Train Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =====================================
# 6. Train Baseline Models
# =====================================

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)


# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)


print("\nBaseline Model Accuracy")
print("Logistic Regression:", lr_acc)
print("Decision Tree:", dt_acc)


# =====================================
# 7. Convert Dataset for Fairness Metrics
# =====================================

# Need original protected attribute
df_original = pd.read_csv("adult.csv")

df_original.replace("?", np.nan, inplace=True)
df_original.dropna(inplace=True)

# Encode income manually
df_original["income"] = df_original["income"].apply(lambda x: 1 if ">50K" in x else 0)

# Encode sex
df_original["sex"] = df_original["sex"].apply(lambda x: 1 if x.strip()=="Male" else 0)

dataset = BinaryLabelDataset(
    df=df_original,
    label_names=["income"],
    protected_attribute_names=["sex"]
)


# =====================================
# 8. Compute Disparate Impact
# =====================================

metric = BinaryLabelDatasetMetric(
    dataset,
    privileged_groups=[{"sex":1}],
    unprivileged_groups=[{"sex":0}]
)

dir_before = metric.disparate_impact()

print("\nDisparate Impact BEFORE mitigation:", dir_before)


# =====================================
# 9. Apply Disparate Impact Remover
# =====================================

dir_remover = DisparateImpactRemover(repair_level=1.0)

dataset_repaired = dir_remover.fit_transform(dataset)

df_repaired = dataset_repaired.convert_to_dataframe()[0]


# =====================================
# 10. Retrain Models on Repaired Data
# =====================================

# Prepare repaired dataset
X_r = df_repaired.drop("income", axis=1)
y_r = df_repaired["income"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_r, y_r, test_size=0.2, random_state=42
)


# Logistic Regression
lr2 = LogisticRegression(max_iter=1000)
lr2.fit(X_train_r, y_train_r)
lr_pred_r = lr2.predict(X_test_r)
lr_acc_r = accuracy_score(y_test_r, lr_pred_r)


# Decision Tree
dt2 = DecisionTreeClassifier()
dt2.fit(X_train_r, y_train_r)
dt_pred_r = dt2.predict(X_test_r)
dt_acc_r = accuracy_score(y_test_r, dt_pred_r)


print("\nAfter Bias Mitigation")
print("Logistic Regression Accuracy:", lr_acc_r)
print("Decision Tree Accuracy:", dt_acc_r)


# =====================================
# 11. Compute DIR After Mitigation
# =====================================

dataset2 = BinaryLabelDataset(
    df=df_repaired,
    label_names=["income"],
    protected_attribute_names=["sex"]
)

metric2 = BinaryLabelDatasetMetric(
    dataset2,
    privileged_groups=[{"sex":1}],
    unprivileged_groups=[{"sex":0}]
)

dir_after = metric2.disparate_impact()

print("\nDisparate Impact AFTER mitigation:", dir_after)


# =====================================
# 12. Accuracy Comparison Graph
# =====================================

models = ["Logistic Regression", "Decision Tree"]

accuracy_before = [lr_acc, dt_acc]
accuracy_after = [lr_acc_r, dt_acc_r]

x = np.arange(len(models))

plt.bar(x - 0.2, accuracy_before, width=0.4, label="Before DIR")
plt.bar(x + 0.2, accuracy_after, width=0.4, label="After DIR")

plt.xticks(x, models)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Before vs After Bias Mitigation")
plt.legend()

plt.show()


# =====================================
# 13. Disparate Impact Graph
# =====================================

labels = ["Before Mitigation", "After Mitigation"]
values = [dir_before, dir_after]

plt.bar(labels, values)

plt.ylabel("Disparate Impact")
plt.title("Disparate Impact Comparison")

plt.show()