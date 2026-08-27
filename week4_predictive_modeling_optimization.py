import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("week4_logistics_prediction_dataset.csv")

features = [
    "distance_km", "weight_kg", "volume_units",
    "transport_mode", "region", "priority"
]
X = pd.get_dummies(df[features], drop_first=True, dtype=float)
y = df["delivery_time_days"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42
    )
}

for name, model in models.items():
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    mae = mean_absolute_error(y_test, prediction)
    rmse = np.sqrt(mean_squared_error(y_test, prediction))
    r2 = r2_score(y_test, prediction)

    print(f"\n{name}")
    print("MAE:", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R2:", round(r2, 3))

# Simple optimization strategy:
# For each shipment, compare transport modes using:
# estimated delivery time + transportation cost penalty.
# Faster modes are reserved for high-priority/high-risk shipments.
