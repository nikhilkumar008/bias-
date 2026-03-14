import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# -------------------------
# 1. Load and Clean Dataset
# -------------------------
# Downloaded from: https://www.kaggle.com/datasets/uciml/adult-census-income
df = pd.read_csv("adult.csv")

# replace '?' with NaN and drop
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)

# Convert Target and Protected Attribute to Binary
df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})

# -------------------------
# 2. Feature Engineering
# -------------------------
X = df.drop(columns=['income'])
y = df['income']

# Identify categorical columns
categorical_cols = ['workclass', 'education', 'marital.status', 
                    'occupation', 'relationship', 'race', 'native.country']

# One-hot encode
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# -------------------------
# 3. Train-Test Split & Scaling
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling is essential for Logistic Regression convergence
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# 4. Train Baseline Models
# -------------------------
# Logistic Regression
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train) # Trees don't strictly require scaling
dt_pred = dt.predict(X_test)

# -------------------------
# 5. Calculate Bias (Disparate Impact)
# -------------------------
def get_metrics(X_test_df, predictions, model_name):
    # We use the original X_test (not scaled) to easily filter by 'sex'
    temp_df = X_test_df.copy()
    temp_df['pred'] = predictions
    
    # Selection Rate = % of people predicted to earn >50K
    sr_male = temp_df[temp_df['sex'] == 1]['pred'].mean()
    sr_female = temp_df[temp_df['sex'] == 0]['pred'].mean()
    
    di = sr_female / sr_male
    acc = accuracy_score(y_test, predictions)
    
    print(f"\n[{model_name} Results]")
    print(f"Accuracy: {acc:.4f}")
    print(f"Selection Rate (Male): {sr_male:.4f}")
    print(f"Selection Rate (Female): {sr_female:.4f}")
    print(f"Disparate Impact (DI): {di:.4f}")
    return di, acc

lr_di, lr_acc = get_metrics(X_test, lr_pred, "Logistic Regression")
dt_di, dt_acc = get_metrics(X_test, dt_pred, "Decision Tree")

# -------------------------
# 6. Result Visualization
# -------------------------
labels = ['Logistic Regression', 'Decision Tree']
dis = [lr_di, dt_di]

plt.figure(figsize=(8, 5))
plt.bar(labels, dis, color=['skyblue', 'salmon'])
plt.axhline(y=0.8, color='red', linestyle='--', label='Fairness Threshold (0.8)')
plt.ylabel('Disparate Impact Score')
plt.title('Baseline Bias Before DIR (Lower than 0.8 is Biased)')
plt.ylim(0, 1.1)
plt.legend()
plt.show()