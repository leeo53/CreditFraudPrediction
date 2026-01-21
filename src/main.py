import pandas as pd
import random_forest
import gradient_boosting
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
import pickle

train_random_forest = False
train_gradient_boosting = True

datapath = Path(__file__).parent.parent / "data"
df = pd.read_csv(datapath / 'creditcard.csv')

X = df.drop(columns=['Class']).values
y = df['Class'].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

fraud_idx = np.where(y_train == 1)[0]
normal_idx = np.where(y_train == 0)[0]

normal_sample = np.random.choice(normal_idx, size=20000, replace=False)

subset_idx = np.concatenate([fraud_idx, normal_sample])

X_train_sub = X_train[subset_idx]
y_train_sub = y_train[subset_idx]

X_train_sub = X_train_sub.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train_sub = y_train_sub.astype(np.int8)
y_test = y_test.astype(np.int8)

rf_model = random_forest.RandomForest(X_train_sub, y_train_sub, 50, 10, 10)
gb_model = gradient_boosting.GradientBoosting(X_train_sub, y_train_sub, 100, 0.1, 5)
if train_random_forest:
    rf_model.fit(4)
    # Save to a file
    with open("rf_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)
else:
    with open("rf_model.pkl", "rb") as f:
        rf_model = pickle.load(f)

if train_gradient_boosting:
    gb_model.fit()
    with open("gb_model.pkl", "wb") as f:
        pickle.dump(gb_model, f)
else:
    with open("gb_model.pkl", "rb") as f:
        gb_model = pickle.load(f)

y_pred_rf = rf_model.predict(X_test)

y_pred_gb, probabilities = gb_model.predict(X_test)
print(probabilities)
thresholds = np.linspace(0, 1, 101)
best_f1 = 0
best_thresh = 0
for t in thresholds:
    y_pred = (probabilities > t).astype(int)
    f1 = f1_score(y_test, y_pred_gb)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print("Best threshold:", best_thresh, "F1:", best_f1)
print("Random Forest:")
print(classification_report(y_test, y_pred_rf, digits=4))
print("Gradient Boosting:")
print(classification_report(y_test, y_pred_gb, digits=4))