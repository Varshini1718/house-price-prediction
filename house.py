import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib

base_dir = Path(__file__).resolve().parent
candidates = [
    base_dir / "data" / "housing.csv",
    base_dir / "data" / "data.csv",
    base_dir / "archive (3)" / "data.csv"
]

dataset_path = next((path for path in candidates if path.exists()), None)
if dataset_path is None:
    raise FileNotFoundError(
        "Could not find housing dataset. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )

print(f"Loading data from: {dataset_path}")
df = pd.read_csv(dataset_path)

df.head()
print(df.shape)

print(df.info())

print(df.describe())
df.isnull().sum().sort_values(ascending=False)

possible_targets = ["SalePrice", "price"]
target = next((col for col in possible_targets if col in df.columns), None)
if target is None:
    raise KeyError(
        "Could not determine target column. Expected one of: "
        + ", ".join(possible_targets)
    )
print(f"Using target column: {target}")

plt.figure(figsize=(8,5))

sns.histplot(df[target], kde=True)

plt.title("House Price Distribution")
plt.show()
df[target] = np.log1p(df[target])
sns.histplot(df[target], kde=True)
plt.title(f"Log Transformed {target}")
plt.show()
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(12,8))

sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()
X = df.drop(target, axis=1)

y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        random_state=42
    )
}
results = {}

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results[name] = {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "pipeline": pipeline
    }

    print(f"\n{name}")
    print("-" * 40)

    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2:", r2)

best_model_name = min(
    results,
    key=lambda x: results[x]["RMSE"]
)

best_pipeline = results[best_model_name]["pipeline"]

print("Best Model:", best_model_name)
preds = best_pipeline.predict(X_test)

residuals = y_test - preds
plt.figure(figsize=(8,5))

sns.scatterplot(x=preds, y=residuals)

plt.axhline(0, color='red', linestyle='--')

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")

plt.title("Residual Analysis")
plt.show()
sns.histplot(residuals, kde=True)

plt.title("Residual Distribution")
plt.show()

output_dir = base_dir / "models"
output_dir.mkdir(exist_ok=True)
joblib.dump(
    best_pipeline,
    output_dir / "best_house_model.pkl"
)

print(f"Model saved successfully: {output_dir / 'best_house_model.pkl'}")