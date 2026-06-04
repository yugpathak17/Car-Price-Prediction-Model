import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

df = pd.read_csv(
    r"C:\Users\YUG M PATHAK\PyCharmMiscProject\codealpha projects\car data.csv"
)

print("First 5 rows of data:")
print(df.head())

print("\nInfo:")
print(df.info())

print("\nDescription:")
print(df.describe(include="all"))

df = df.drop_duplicates()

df = df.dropna(subset=["Selling_Price"])

df = df.dropna()

print(f"\nShape after cleaning: {df.shape}")

CURRENT_YEAR = 2026

if "Year" in df.columns:
    df["car_age"] = CURRENT_YEAR - df["Year"]
    df = df.drop(columns=["Year"])

if "Car_Name" in df.columns:
    df = df.drop(columns=["Car_Name"])

target_col = "Selling_Price"

X = df.drop(columns=[target_col])

y = df[target_col]

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["category", "bool"]
).columns.tolist()

categorical_features += X.select_dtypes(
    include="object"
).columns.tolist()

print("\nNumeric features:", numeric_features)

print("Categorical features:", categorical_features)

plt.figure()

sns.histplot(df[target_col], kde=True)

plt.title("Car Selling Price Distribution")

plt.xlabel("Selling Price (in lakhs)")

plt.ylabel("Count")

plt.tight_layout()

plt.show()

if "Driven_kms" in df.columns:
    plt.figure()

    sns.scatterplot(
        data=df,
        x="Driven_kms",
        y=target_col
    )

    plt.title("Driven KMs vs Selling Price")

    plt.tight_layout()

    plt.show()

if len(numeric_features) > 1:
    plt.figure(figsize=(10, 8))

    corr = df[numeric_features + [target_col]].corr()

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Heatmap (Numeric Features)")

    plt.tight_layout()

    plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        ),
    ]
)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
)

regressor = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        ),
    ]
)

print("\nTraining model...")

regressor.fit(X_train, y_train)

print("Training complete.\n")

y_pred = regressor.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print(f"MAE : {mae:.2f}")

print(f"RMSE: {rmse:.2f}")

print(f"R²  : {r2:.3f}")

plt.figure()

sns.scatterplot(
    x=y_test,
    y=y_pred
)

plt.xlabel("Actual Selling Price")

plt.ylabel("Predicted Selling Price")

plt.title("Actual vs Predicted Car Prices")

min_val = min(y_test.min(), y_pred.min())

max_val = max(y_test.max(), y_pred.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    "r--"
)

plt.tight_layout()

plt.show()

preprocessor_fitted = regressor.named_steps["preprocessor"]

feature_names_num = numeric_features

if categorical_features:
    ohe = preprocessor_fitted.named_transformers_["cat"]

    feature_names_cat = ohe.get_feature_names_out(
        categorical_features
    ).tolist()

else:
    feature_names_cat = []

all_feature_names = (
    feature_names_num +
    feature_names_cat
)

rf_model = regressor.named_steps["model"]

importances = rf_model.feature_importances_

feat_imp = pd.Series(
    importances,
    index=all_feature_names
).sort_values(ascending=False)

print("\nTop 15 important features:")

print(feat_imp.head(15))

plt.figure(figsize=(10, 6))

sns.barplot(
    x=feat_imp.head(15),
    y=feat_imp.head(15).index
)

plt.title("Top 15 Feature Importances")

plt.xlabel("Importance Score")

plt.ylabel("Feature")

plt.tight_layout()

plt.show()

MODEL_PATH = "car_price_model.pkl"

joblib.dump(regressor, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")

example_dict = {}

for col in X.columns:
    if X[col].dtype in ["int64", "float64"]:
        example_dict[col] = X[col].median()

    else:
        example_dict[col] = X[col].mode()[0]

new_car = pd.DataFrame([example_dict])

print("\nExample input for a new car:")

print(new_car)

predicted_price = regressor.predict(new_car)[0]

print(
    f"\nPredicted Selling Price for example car: "
    f"{predicted_price:.2f} lakhs"
)