import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

## SETUP
os.makedirs("outputs/results", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

## LOAD DATA
df = pd.read_parquet("order_products.parquet")
df = df.sample(frac=0.2, random_state=42)

## DEFINE COLS
USER_COL = "user_id"
PRODUCT_COL = "product_id"
TARGET_COL = "reordered"   # must be 0/1

## BASIC CLEANING
df = df.dropna().drop_duplicates()


## FEATURE ENGG
# Product popularity
product_freq = df[PRODUCT_COL].value_counts()
df["product_freq"] = df[PRODUCT_COL].map(product_freq)

# User activity
user_orders = df.groupby(USER_COL).size()
df["user_total_orders"] = df[USER_COL].map(user_orders)

# User-product interaction
user_product = df.groupby([USER_COL, PRODUCT_COL]).size()
df["user_product_count"] = df.set_index([USER_COL, PRODUCT_COL]).index.map(user_product)

# Encode product as category (efficient)
df[PRODUCT_COL] = df[PRODUCT_COL].astype("category").cat.codes

## FEATURE MATRIX
X = df[[PRODUCT_COL, "product_freq", "user_total_orders", "user_product_count"]]
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


## XG BOost
xgb = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist"   # 🔥 faster for large data
)

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)


## ANN
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train_scaled, y_train, epochs=10, batch_size=64, verbose=0)

y_pred_ann = (model.predict(X_test_scaled) > 0.5).astype(int)

## EVAL
def evaluate(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred)
    }

xgb_results = evaluate(y_test, y_pred_xgb)
ann_results = evaluate(y_test, y_pred_ann)

## SAVE RESULTS
results_df = pd.DataFrame([xgb_results, ann_results],
                          index=["XGBoost", "ANN"])

results_df.to_csv("outputs/results/model_results.csv")

print(results_df)

## PLOT Comp
results_df.plot(kind='bar')
plt.title("Model Comparison")
plt.xticks(rotation=0)
plt.savefig("outputs/plots/model_comparison.png")
plt.close()