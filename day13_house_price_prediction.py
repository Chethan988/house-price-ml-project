# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# =========================
# 2. LOAD DATASET
# =========================
data = fetch_california_housing()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["price"] = data.target

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())


# =========================
# 3. EDA (Visualization)
# =========================
sns.histplot(df["price"], bins=30)
plt.title("Price Distribution")
plt.show()

corr = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


# =========================
# 4. SPLIT DATA
# =========================
X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# 5. LINEAR REGRESSION (PIPELINE)
# =========================
pipeline_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

pipeline_lr.fit(X_train, y_train)
pred_lr = pipeline_lr.predict(X_test)

print("\n--- Linear Regression (Pipeline) ---")
print("MSE:", mean_squared_error(y_test, pred_lr))
print("R2:", r2_score(y_test, pred_lr))


# =========================
# 6. RANDOM FOREST MODEL
# =========================
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)
pred_rf = rf_model.predict(X_test)

print("\n--- Random Forest ---")
print("MSE:", mean_squared_error(y_test, pred_rf))
print("R2:", r2_score(y_test, pred_rf))


# =========================
# 7. MODEL COMPARISON
# =========================
print("\n----- MODEL COMPARISON -----")
print("Linear Regression R2:", r2_score(y_test, pred_lr))
print("Random Forest R2:", r2_score(y_test, pred_rf))


# =========================
# 8. FEATURE IMPORTANCE (Random Forest)
# =========================
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n--- Feature Importance ---")
print(importance)


# =========================
# 9. COEFFICIENTS (Linear Regression)
# =========================
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": pipeline_lr.named_steps["model"].coef_
})

print("\n--- Linear Regression Coefficients ---")
print(coefficients)